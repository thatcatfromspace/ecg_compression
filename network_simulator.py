import time
import json
import logging
from pathlib import Path
from typing import List

# Configure logging
logger = logging.getLogger(__name__)


class ECGNetworkSimulator:
    """
    Simulates network transmission by writing chunks to separate files
    """

    def __init__(
        self, output_dir: str = "output", transmission_delay: float = 0.001
    ):  # 1ms delay per chunk
        """
        Initialize network simulator

        Args:
            output_dir: Directory to write transmitted chunks
            transmission_delay: Simulated network delay per chunk in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.transmission_delay = transmission_delay
        self.transmitted_chunks = []

    def transmit_chunk(self, chunk_data: bytes, chunk_id: int, crc32: int) -> bool:
        """
        Simulate chunk transmission by writing to file

        Args:
            chunk_data: Compressed chunk data
            chunk_id: Unique chunk identifier
            crc32: CRC32 checksum of the chunk

        Returns:
            True if transmission successful
        """
        try:
            # Simulate network delay
            time.sleep(self.transmission_delay)

            # Create chunk file
            chunk_file = self.output_dir / f"chunk_{chunk_id:06d}.bin"

            # Write chunk with metadata
            chunk_packet = {
                "chunk_id": chunk_id,
                "crc32": crc32,
                "data_size": len(chunk_data),
                "timestamp": time.time(),
            }

            with open(chunk_file, "wb") as f:
                # Write metadata header (JSON)
                header = json.dumps(chunk_packet).encode("utf-8")
                header_size = len(header)

                # Write: header_size (4 bytes) + header + data
                f.write(header_size.to_bytes(4, byteorder="little"))
                f.write(header)
                f.write(chunk_data)

            self.transmitted_chunks.append(chunk_id)
            return True

        except Exception as e:
            logger.error(f"Transmission failed for chunk {chunk_id}: {e}")
            return False

    def get_transmitted_chunks(self) -> List[int]:
        """Get list of successfully transmitted chunk IDs"""
        return sorted(self.transmitted_chunks)
