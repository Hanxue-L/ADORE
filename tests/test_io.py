from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from adore_dpv.io import read_dpv_trace


class DPVInputTest(unittest.TestCase):
    def test_reads_headerless_numeric_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.txt"
            path.write_text("0.0 1.0\n0.1 2.0\n0.2 1.0\n", encoding="utf-8")

            observed = read_dpv_trace(path)

        self.assertEqual(observed["voltage"].tolist(), [0.0, 0.1, 0.2])
        self.assertEqual(observed["current"].tolist(), [1.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
