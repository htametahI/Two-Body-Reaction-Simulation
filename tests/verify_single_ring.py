import csv
import json
import sys
import math
from pathlib import Path
import uproot

root = Path(sys.argv[1])
cases = [[], [1], [1, 5], [1, 1], [15], [1, 15],
         [1, 5, 15], [23], [15, 16], [], [0], [1, 16, 17]]
observed = {}
for line in (root / 'observed.txt').read_text().splitlines():
    if line.startswith('OBSERVED '):
        _, event, *values = line.split()
        observed[int(event)] = {int(v.split(':')[0]): int(v.split(':')[1]) for v in values}
assert len(observed) == len(cases), observed
for event, rings in enumerate(cases):
    assert set(observed[event]) == set(rings), (event, observed[event], rings)
assert observed[3][1] >= 2, 'Repeated deposits in one ring were not exercised'

# Expected accepted event IDs and one-based CSV ring numbers.
expected = {1: 2, 3: 2, 5: 2, 7: 24, 10: 1, 11: 2}
with (root / 'check_exit.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len(cases), len(rows)
selected = {}
for row in rows:
    event = int(row['eventID'])
    enabled_count = len(set(cases[event]) - {15, 16, 17})
    assert int(row['nS3EnabledRingsHit']) == enabled_count
    assert int(row['passesS3Selection']) == (enabled_count == 1)
    ring = float(row['triton_S3_ringID'])
    if math.isfinite(ring):
        selected[event] = int(ring)
    else:
        for column in ['triton_S3_edep_smeared_MeV', 'triton_E_reco_center_MeV', 'beam_E_center_MeV']:
            assert math.isnan(float(row[column])), (event, column)
assert selected == expected, selected

with (root / 'check_transmission.csv').open() as stream:
    assert len(list(csv.DictReader(stream))) == len(expected) - 1
with uproot.open(root / 'check.root') as output:
    hits = output['S3'].arrays(library='np')
    pairs = list(zip(map(int, hits['eventID']), map(int, hits['ringID'])))
    assert pairs == [(event, ring - 1) for event, ring in expected.items()], pairs
    assert output['reaction'].num_entries == len(cases)
    reaction = output['reaction'].arrays(library='np')
    assert reaction['nS3EnabledRingsHit'].tolist() == [0, 1, 2, 1, 0, 1, 2, 1, 0, 0, 1, 1]
    for row in rows:
        event = int(row['eventID'])
        for flag, energy in [('hasMg26Exit', 'Mg26_E_MeV'), ('hasTritonExit', 'triton_E_MeV')]:
            assert int(row[flag]) == reaction[flag][event]
            if not int(row[flag]):
                assert math.isnan(float(row[energy])) and math.isnan(reaction[energy][event])
    for hist in ['hS3_edep', 'hS3_total_edep', 'hS3_edep_vs_ring']:
        assert output[hist].values(flow=True).sum() == len(expected), hist
print('PASS: 12 controlled events; exactly 6 single-enabled-ring events selected.')
print('PASS: zero/multiple enabled rings rejected; repeated deposits and excluded-ring combinations handled correctly.')
print('PASS: ROOT hits/histograms, event CSV S3 fields, transmission count, and event resets agree.')

metadata = json.loads((root / 'check_metadata.json').read_text())
assert metadata['status'] == 'complete'
assert metadata['run_id'] == 1
assert metadata['s3_event_counts'] == {'zero_enabled_rings': 4, 'one_enabled_ring': 6, 'multiple_enabled_rings': 2}
assert metadata['selection']['excluded_ring_numbers'] == [16, 17, 18]
for boundary in ['start_state', 'end_state']:
    assert Path(metadata['random'][boundary]).stat().st_size > 0
print('PASS: repeated-run reset, missing-exit flags/NaNs, selection metadata, and RNG states.')
