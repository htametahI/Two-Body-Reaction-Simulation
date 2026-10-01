"""Check beam provenance without running plotting or modifying user outputs."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Plotting"))
from simulation_io import read_event_beam, _fingerprint


class BeamProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_embedded_state_survives_replaced_beam(self):
        (self.root / "state_truth_kinematics.csv").write_text(
            "eventID,beam_E_MeV,beam_x_mm,beam_y_mm,beam_z_mm,beam_dirx,beam_diry,beam_dirz,depth_um\n"
            "7,64,0,0,0.251,0,0,1,1.2\n"
        )
        (self.root / "beam.dat").write_text("10 0 0 0 0 0 1 0.1\n")
        actual = read_event_beam(self.root, "state")
        self.assertEqual(actual["eventID"].tolist(), [7])
        self.assertEqual(actual["depth_um"].tolist(), [1.2])
        (self.root / "beam.dat").unlink()
        self.assertEqual(read_event_beam(self.root, "state")["Ebeam_MeV"].tolist(), [64])

    def test_legacy_fallback_and_identity(self):
        (self.root / "state_truth_kinematics.csv").write_text("eventID,triton_E_MeV\n0,1\n")
        beam = self.root / "beam.dat"
        beam.write_text("64 0 0 0 0 0 1 0.4\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(read_event_beam(self.root, "state")["depth_um"].tolist(), [0.4])
            self.assertTrue(caught)
        size, fingerprint = _fingerprint(beam)
        (self.root / "state_metadata.json").write_text(json.dumps({"inputs": {"beam": {
            "path": str(beam), "size_bytes": size, "fnv1a64": fingerprint,
        }}}))
        self.assertEqual(len(read_event_beam(self.root, "state")), 1)
        beam.write_text("65 0 0 0 0 0 1 0.5\n")
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            read_event_beam(self.root, "state")

    def test_partial_state_is_rejected(self):
        (self.root / "state_truth_kinematics.csv").write_text("eventID,depth_um\n0,0.5\n")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            read_event_beam(self.root, "state")


if __name__ == "__main__":
    unittest.main()
