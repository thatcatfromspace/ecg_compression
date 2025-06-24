import logging
import queue
import threading
import time
import zlib
from typing import Tuple

import boto3
import numpy as np
import wfdb
import zstandard as zstd

from network_simulator import ECGNetworkSimulator

logger = logging.getLogger(__name__)


class ECGStreamingCompressor:
    """
    Enhanced ECG streaming compressor with network simulation and CRC32
    """

    def __init__(
        self,
        sampling_rate: int = 25000,
        chunk_size: int = 1000,
        quantization_scale: int = 1000,
        compression_level: int = 3,
        simulate_network: bool = True,
        output_dir: str = "output",
    ):
        """
        Initialize the ECG streaming compressor
        """
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.quantization_scale = quantization_scale
        self.compression_level = compression_level
        self.simulate_network = simulate_network

        # Streaming control
        self.data_queue = queue.Queue(maxsize=100)
        self.compressed_data = []
        self.is_streaming = False
        self.total_samples = 0
        self.processed_samples = 0

        # Network simulation
        if simulate_network:
            self.network_sim = ECGNetworkSimulator(output_dir)
        else:
            self.network_sim = None

        # Initialize compressor and decompressor
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.decompressor = zstd.ZstdDecompressor()

        # AWS S3 client (optional)
        try:
            self.s3_client = boto3.client("s3")
        except Exception as e:
            self.s3_client = None
            logger.warning(f"AWS S3 client not available: {e}")

    def load_ecg_data(self, record_path: str) -> Tuple[np.ndarray, dict]:
        """Load ECG data using wfdb"""
        try:
            record = wfdb.rdrecord(record_path)
            signal_data = record.p_signal

            metadata = {
                "fs": record.fs,
                "sig_len": len(signal_data),
                "n_sig": record.n_sig,
                "sig_name": record.sig_name,
                "units": record.units,
            }

            logger.info(f"Loaded ECG data: {signal_data.shape}, Fs={metadata['fs']}Hz")
            return signal_data, metadata

        except Exception as e:
            logger.error(f"Error loading ECG data: {e}")
            raise

    def quantize_data(self, data: np.ndarray) -> np.ndarray:
        """Quantize ECG data to integers"""
        quantized = np.round(data * self.quantization_scale).astype(np.int32)
        return quantized

    def compress_chunk(self, chunk: np.ndarray) -> Tuple[bytes, int]:
        """
        Compress a chunk and calculate CRC32

        Returns:
            Tuple of (compressed_data, crc32)
        """
        quantized = self.quantize_data(chunk)

        delta_array = np.vstack(
            [quantized[0:1], np.diff(quantized, axis=0)]  # keep first sample as-is
        )

        # Convert to bytes
        chunk_bytes = delta_array.tobytes()

        # Compress
        compressed = self.compressor.compress(chunk_bytes)

        # Calculate CRC32
        crc32 = zlib.crc32(compressed) & 0xFFFFFFFF

        assert delta_array.shape == chunk.shape

        return compressed, crc32

    def decompress_chunk(
        self, compressed_data: bytes, original_shape: tuple
    ) -> np.ndarray:
        """
        Decompress a chunk back to original data

        Args:
            compressed_data: Compressed chunk data
            original_shape: Original shape of the chunk

        Returns:
            Decompressed numpy array
        """
        try:
            # Decompress
            decompressed_bytes = self.decompressor.decompress(compressed_data)

            # Convert back to numpy array
            delta_array = np.frombuffer(decompressed_bytes, dtype=np.int32).reshape(
                original_shape
            )
            reconstructed_array = np.cumsum(delta_array, axis=0)

            # Dequantize
            original_data = (
                reconstructed_array.astype(np.float64) / self.quantization_scale
            )

            assert reconstructed_array.shape == original_shape

            return original_data

        except Exception as e:
            logger.error(f"Error decompressing chunk: {e}")
            raise

    def data_producer(self, signal_data: np.ndarray):
        """Producer thread for streaming data"""
        self.total_samples = len(signal_data)
        sample_interval = 1.0 / self.sampling_rate
        chunk_interval = self.chunk_size * sample_interval

        logger.info(f"Starting data streaming at {self.sampling_rate}Hz")

        start_time = time.time()
        chunk_count = 0

        for i in range(0, len(signal_data), self.chunk_size):
            if not self.is_streaming:
                break

            end_idx = min(i + self.chunk_size, len(signal_data))
            chunk = signal_data[i:end_idx]

            # Timing control
            expected_time = start_time + (chunk_count * chunk_interval)
            current_time = time.time()
            sleep_time = expected_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            try:
                self.data_queue.put((chunk, chunk_count, i), timeout=1.0)
                chunk_count += 1

                if chunk_count % 100 == 0:
                    progress = (i / len(signal_data)) * 100
                    logger.info(f"Streaming progress: {progress:.1f}%")

            except queue.Full:
                logger.warning("Data queue full, dropping chunk")

        self.data_queue.put((None, -1, -1))
        logger.info("Data streaming completed")

    def data_consumer(self):
        """Consumer thread for processing and transmitting data"""
        logger.info("Starting data compression and transmission")

        while self.is_streaming:
            try:
                chunk, chunk_id, start_idx = self.data_queue.get(timeout=1.0)

                if chunk is None:
                    break

                # Quantize and compress
                compressed_chunk, crc32 = self.compress_chunk(chunk)

                # Store chunk info
                chunk_info = {
                    "chunk_id": chunk_id,
                    "data": compressed_chunk,
                    "original_shape": chunk.shape,
                    "start_idx": start_idx,
                    "crc32": crc32,
                    "compression_ratio": chunk.nbytes / len(compressed_chunk),
                }

                self.compressed_data.append(chunk_info)

                # Simulate network transmission
                if self.network_sim:
                    transmission_success = self.network_sim.transmit_chunk(
                        compressed_chunk, chunk_id, crc32
                    )
                    chunk_info["transmitted"] = transmission_success

                self.processed_samples += len(chunk)

                # Progress logging
                if len(self.compressed_data) % 100 == 0:
                    avg_ratio = np.mean(
                        [c["compression_ratio"] for c in self.compressed_data]
                    )
                    logger.info(
                        f"Processed {len(self.compressed_data)} chunks, "
                        f"avg compression ratio: {avg_ratio:.3f}"
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in data consumer: {e}")
                break

        logger.info(
            f"Data processing completed. Processed {self.processed_samples} samples"
        )

    def stream_and_compress(self, record_path: str) -> dict:
        """Main streaming and compression function"""
        # Load ECG data
        signal_data, metadata = self.load_ecg_data(record_path)

        # Clean output directory if simulating network
        if self.network_sim:
            import shutil

            if self.network_sim.output_dir.exists():
                shutil.rmtree(self.network_sim.output_dir)
            self.network_sim.output_dir.mkdir(exist_ok=True)

        # Initialize streaming
        self.is_streaming = True
        self.compressed_data = []
        self.processed_samples = 0

        # Start threads
        producer_thread = threading.Thread(
            target=self.data_producer, args=(signal_data,)
        )
        consumer_thread = threading.Thread(target=self.data_consumer)

        start_time = time.time()

        producer_thread.start()
        consumer_thread.start()

        # Wait for completion
        producer_thread.join()
        consumer_thread.join()

        self.is_streaming = False

        # Calculate statistics
        end_time = time.time()
        processing_time = end_time - start_time

        original_size = signal_data.nbytes
        compressed_size = sum(len(chunk["data"]) for chunk in self.compressed_data)

        stats = {
            "original_size_mb": original_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": original_size / compressed_size,
            "processing_time_sec": processing_time,
            "throughput_samples_per_sec": self.processed_samples / processing_time,
            "num_chunks": len(self.compressed_data),
            "metadata": metadata,
            "original_data": signal_data,  # Store for comparison
        }

        return stats

    def test_transmission_and_recovery(self, output_dir: str = "output") -> dict:
        """
        Test transmission simulation and data recovery

        Returns:
            Recovery test results
        """
        from receiver import ECGReceiver

        if not self.network_sim:
            logger.error("Network simulation not enabled")
            return {}

        logger.info("Testing transmission and recovery...")

        # Simulate receiving data
        receiver = ECGReceiver(output_dir)
        reception_stats = receiver.receive_all_chunks()

        # Get ordered chunks
        received_chunks = receiver.get_ordered_chunks()

        # Decompress all chunks
        recovered_data = []
        decompression_errors = 0

        for i, chunk_data in enumerate(received_chunks):
            try:
                # Find original chunk info
                original_chunk = None
                for chunk_info in self.compressed_data:
                    if chunk_info["chunk_id"] == i:
                        original_chunk = chunk_info
                        break

                if original_chunk:
                    decompressed = self.decompress_chunk(
                        chunk_data, original_chunk["original_shape"]
                    )
                    recovered_data.append(decompressed)
                else:
                    decompression_errors += 1

            except Exception as e:
                logger.error(f"Decompression error for chunk {i}: {e}")
                decompression_errors += 1

        # Reconstruct complete signal
        if recovered_data:
            recovered_signal = np.vstack(recovered_data)
        else:
            recovered_signal = np.array([])

        test_results = {
            "reception_stats": reception_stats,
            "decompression_errors": decompression_errors,
            "recovered_samples": (
                len(recovered_signal) if len(recovered_signal.shape) > 1 else 0
            ),
            "recovery_success_rate": (
                len(received_chunks) / len(self.compressed_data)
                if self.compressed_data
                else 0
            ),
            "recovered_data": recovered_signal,
        }

        return test_results
