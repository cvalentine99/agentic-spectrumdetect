import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import spectrum_server as spectrum_server_pkg

# Ensure the namespace package can find modules under src.
spectrum_server_pkg.__path__.append(str(PROJECT_ROOT / "src" / "spectrum_server"))

from spectrum_server.spectrum_core_client import SpectrumCoreClient


class SpectrumCoreClientContractTest(unittest.TestCase):
    def test_measure_spectrum_parses_reply_and_alias(self):
        header = {
            "op": "measure",
            "schema_version": 1,
            "center_freq_hz": 2_450_000_000,
            "sample_rate_hz": 50_000_000,
            "fft_size": 8,
            "averaging": 4,
            "timestamp_ns": 123456789,
            "detections": [{"detection_id": "abc", "freq_hz": 2_450_000_000, "power_dbm": -50.0}],
        }

        magnitudes = np.arange(header["fft_size"], dtype=np.float32)
        reply_frames = [json.dumps(header).encode("utf-8"), magnitudes.tobytes()]

        client = SpectrumCoreClient(endpoint="inproc://test")
        frame = client._decode_reply(  # type: ignore[attr-defined]
            reply_frames,
            default_center=header["center_freq_hz"],
            default_rate=header["sample_rate_hz"],
            fft_size=header["fft_size"],
            averaging=header["averaging"],
        )

        # Validate the parsed frame mirrors the header and payload.
        self.assertIsNotNone(frame)
        self.assertEqual(frame.center_freq_hz, header["center_freq_hz"])
        self.assertEqual(frame.sample_rate_hz, header["sample_rate_hz"])
        self.assertEqual(frame.fft_size, header["fft_size"])
        self.assertEqual(frame.timestamp_ns, header["timestamp_ns"])
        self.assertEqual(frame.detections, header["detections"])
        self.assertEqual(frame.schema_version, header["schema_version"])

        # Both magnitudes aliases should be populated and equal.
        self.assertEqual(frame.magnitudes_dbm, magnitudes.tolist())
        self.assertEqual(frame.magnitudes_db, frame.magnitudes_dbm)
