import json
import logging
import time
import zlib
from pathlib import Path
from typing import List, Tuple

# Configure logging
logger = logging.getLogger(__name__)


class ECGReceiver:
    """
    Simulates receiving and reassembling transmitted chunks
    """

    def __init__(self, input_dir: str = "output"):
        """
        Initialize receiver

        Args:
            input_dir: Directory containing transmitted chunks
        """
        self.input_dir = Path(input_dir)
        self.received_chunks = {}
        self.crc_failures = []

    def receive_chunk(self, chunk_file: Path) -> Tuple[bool, dict]:
        """
        Receive and validate a chunk file

        Args:
            chunk_file: Path to chunk file

        Returns:
            Tuple of (success, chunk_info)
        """
        try:
            with open(chunk_file, "rb") as f:
                # Read header size
                header_size = int.from_bytes(f.read(4), byteorder="little")

                # Read header
                header_data = f.read(header_size)
                chunk_info = json.loads(header_data.decode("utf-8"))

                # Read chunk data
                chunk_data = f.read()

                # Verify CRC32
                calculated_crc = zlib.crc32(chunk_data) & 0xFFFFFFFF
                expected_crc = chunk_info["crc32"]

                if calculated_crc != expected_crc:
                    logger.error(
                        f"CRC32 mismatch for chunk {chunk_info['chunk_id']}: "
                        f"expected {expected_crc}, got {calculated_crc}"
                    )
                    self.crc_failures.append(chunk_info["chunk_id"])
                    return False, chunk_info

                # Store received chunk
                chunk_info["data"] = chunk_data
                chunk_info["received_time"] = time.time()
                self.received_chunks[chunk_info["chunk_id"]] = chunk_info

                return True, chunk_info

        except Exception as e:
            logger.error(f"Error receiving chunk {chunk_file}: {e}")
            return False, {}

    def receive_all_chunks(self) -> dict:
        """
        Receive all chunks from input directory

        Returns:
            Reception statistics
        """
        chunk_files = sorted(self.input_dir.glob("chunk_*.bin"))

        logger.info(f"Found {len(chunk_files)} chunks to receive")

        successful_receptions = 0

        for chunk_file in chunk_files:
            success, chunk_info = self.receive_chunk(chunk_file)
            if success:
                successful_receptions += 1

            # Log progress
            if len(self.received_chunks) % 100 == 0:
                logger.info(f"Received {len(self.received_chunks)} chunks")

        stats = {
            "total_chunks": len(chunk_files),
            "successful_receptions": successful_receptions,
            "crc_failures": len(self.crc_failures),
            "failed_chunk_ids": self.crc_failures,
            "reception_rate": (
                successful_receptions / len(chunk_files) if chunk_files else 0
            ),
        }

        return stats

    def get_ordered_chunks(self) -> List[bytes]:
        """
        Get received chunks in order

        Returns:
            List of chunk data in order
        """
        ordered_chunks = []
        chunk_ids = sorted(self.received_chunks.keys())

        for chunk_id in chunk_ids:
            ordered_chunks.append(self.received_chunks[chunk_id]["data"])

        return ordered_chunks
