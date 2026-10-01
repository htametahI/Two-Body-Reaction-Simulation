"""Exercise prepare/reaction, full RNG replay, and CLI failure paths in isolation."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sim, source = map(Path, sys.argv[1:3])
dwba = source / "input/11500_1minus/fort.202"
with tempfile.TemporaryDirectory(prefix="s2223-workflow-") as directory:
    work = Path(directory)
    (work / "build").mkdir()
    beam = work / "output/beam.dat"

    def run(*args, expect_error=None):
        result = subprocess.run([str(sim), *map(str, args)], cwd=work / "build",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if expect_error:
            assert result.returncode != 0 and expect_error in result.stdout, result.stdout[-4000:]
        else:
            assert result.returncode == 0, result.stdout[-4000:]
        return result

    run("--prepare", 40, "--seed", 271828, "--beam-file", beam)
    with beam.open() as stream:
        states = [list(map(float, line.split())) for line in stream if not line.startswith("#")]
    assert len(states) == 40
    preparation = json.loads((work / "output/beam_metadata.json").read_text())
    assert preparation['status'] == 'complete' and preparation['prepared_beam']['size_bytes'] == beam.stat().st_size
    original_beam = beam.read_bytes()

    output = work / "output/state.root"
    run("--reaction", dwba, 11.5, output, 12, "--seed", 314159, "--beam-file", beam)
    truth_path = work / "output/state_truth_kinematics.csv"
    exit_path = work / "output/state_exit_kinematics.csv"
    truth_bytes, exit_bytes = truth_path.read_bytes(), exit_path.read_bytes()
    with truth_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    for index, row in enumerate(rows):
        assert int(row['beamRowID']) == index and int(row['eventID']) == index
        assert float(row['beam_E_MeV']) == states[index][0]
        assert float(row['depth_um']) == states[index][7]
    metadata = json.loads((work / "output/state_metadata.json").read_text())
    assert metadata['status'] == 'complete' and sum(metadata['s3_event_counts'].values()) == 12
    assert metadata['inputs']['beam']['size_bytes'] == len(original_beam)

    run("--reaction", dwba, 11.5, work / "output/replay.root", 12,
        "--rng-state", work / "output/state_random_start.rndm", "--beam-file", beam)
    assert (work / "output/replay_truth_kinematics.csv").read_bytes() == truth_bytes, 'Truth RNG replay mismatch'
    assert (work / "output/replay_exit_kinematics.csv").read_bytes() == exit_bytes, 'Transport RNG replay mismatch'

    run("--reaction", dwba, 11.5, work / "output/too_many.root", 41,
        "--beam-file", beam, expect_error="more reaction events than rows")
    bad_dwba = work / "bad_dwba.dat"
    bad_dwba.write_text("0 1\n90 -2\n180 1\n")
    run("--reaction", bad_dwba, 11.5, work / "output/invalid.root", 1,
        "--beam-file", beam, expect_error="bad_dwba.dat:2:")
    blocked = work / "not_a_directory"
    blocked.write_text("file")
    run("--reaction", dwba, 11.5, blocked / "result.root", 1,
        "--beam-file", beam, expect_error="Cannot create output directory")
    run("--prepare", -1, expect_error="positive integer")
    run("--unknown", expect_error="Unknown command")
    run("--prepare", 40, "--seed", 1234, "--beam-file", beam)
    assert beam.read_bytes() != original_beam
    assert truth_path.read_bytes() == truth_bytes, 'Regenerating beam changed existing truth'
print('PASS: isolated prepare/reaction workflow, beam provenance, exact RNG replay, and failure diagnostics')
