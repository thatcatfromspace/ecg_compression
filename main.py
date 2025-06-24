import logging

import numpy as np

from compressor import ECGStreamingCompressor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_ecg_with_network_simulation(
    record_path: str, output_dir: str = "output", run_recovery_test: bool = True
):
    """
    Complete ECG processing pipeline with network simulation
    """

    # Initialize compressor with network simulation
    compressor = ECGStreamingCompressor(
        sampling_rate=25000,
        chunk_size=1000,
        quantization_scale=1000,
        compression_level=3,
        simulate_network=True,
        output_dir=output_dir,
    )

    try:
        # Stream and compress data
        logger.info("Starting ECG compression with network simulation...")
        compression_stats = compressor.stream_and_compress(record_path)

        # Print compression results
        print("\n" + "=" * 60)
        print("ECG COMPRESSION & TRANSMISSION RESULTS")
        print("=" * 60)
        print(f"Original size: {compression_stats['original_size_mb']:.2f} MB")
        print(f"Compressed size: {compression_stats['compressed_size_mb']:.2f} MB")
        print(f"Compression ratio: {compression_stats['compression_ratio']:.3f}")
        print(f"Space saved: {(1-(1/compression_stats['compression_ratio']))*100:.1f}%")
        print(
            f"Processing time: {compression_stats['processing_time_sec']:.2f} seconds"
        )
        print(
            f"Throughput: {compression_stats['throughput_samples_per_sec']:.0f} samples/sec"
        )
        print(f"Number of chunks transmitted: {compression_stats['num_chunks']}")

        # Test transmission and recovery
        if run_recovery_test:
            print("\n" + "=" * 60)
            print("TRANSMISSION & RECOVERY TEST")
            print("=" * 60)

            recovery_results = compressor.test_transmission_and_recovery(output_dir)

            reception_stats = recovery_results["reception_stats"]
            print(f"Total chunks transmitted: {reception_stats['total_chunks']}")
            print(f"Successful receptions: {reception_stats['successful_receptions']}")
            print(f"CRC32 failures: {reception_stats['crc_failures']}")
            print(f"Reception rate: {reception_stats['reception_rate']*100:.1f}%")
            print(f"Decompression errors: {recovery_results['decompression_errors']}")
            print(
                f"Recovery success rate: {recovery_results['recovery_success_rate']*100:.1f}%"
            )
            print(f"Recovered samples: {recovery_results['recovered_samples']}")

            # Data integrity check
            if len(recovery_results["recovered_data"]) > 0:
                original_data = compression_stats["original_data"]
                recovered_data = recovery_results["recovered_data"]

                # Compare first few samples
                if len(recovered_data) >= len(original_data):
                    max_error = np.max(
                        np.abs(original_data - recovered_data[: len(original_data)])
                    )
                    mean_error = np.mean(
                        np.abs(original_data - recovered_data[: len(original_data)])
                    )

                    print("Data integrity check:")
                    print(f"Max reconstruction error: {max_error:.6f}")
                    print(f"Mean reconstruction error: {mean_error:.6f}")

                    if max_error < 1e-6:
                        print("Perfect reconstruction achieved. (less than 10^-6)")
                    else:
                        print("Some reconstruction errors detected")
                else:
                    print("Incomplete data recovery")

        print("=" * 60)

        # Print sample data of original and recovered data
        if run_recovery_test and len(recovery_results["recovered_data"]) > 0 and len(compression_stats["original_data"]) > 0:
            print("\nSample data:")
            print("Original data (first 10 samples):")
            print(compression_stats["original_data"][:10])
            print("Recovered data (first 10 samples):")
            print(recovery_results["recovered_data"][:10])

        return {
            "compression_stats": compression_stats,
            "recovery_results": recovery_results if run_recovery_test else None,
        }

    except Exception as e:
        logger.error(f"Error in ECG processing pipeline: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        results = process_ecg_with_network_simulation("data/test01_00s")
        print("Processing completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
