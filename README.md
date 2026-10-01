# S2223 simulation

This sequential Geant4 simulation prepares a transported 22Ne beam in a LiF/C
target, then generates 22Ne + 7Li -> 26Mg* + t reactions from a DWBA angular table.
The reaction products are transported through the target and the S3 geometry.

## Build and run

Geant4 with UI/visualization support and a C++17 compiler are required.

```sh
cmake -S . -B build
cmake --build build --parallel
cd build
./sim --prepare 10000 --seed 12345
./sim --reaction ../input/11500_1minus/fort.202 11.5 ../output/11500_1minus.root --seed 67890
```

Without arguments, `sim` performs both stages with 1000 preparation events.
`--vis` also performs both stages and starts the interactive viewer. A macro
filename runs in preparation mode. `./sim --help` lists the options.

- `--beam-file PATH` selects the preparation output / reaction input.
- An optional positional event count follows `--prepare`, or the ROOT filename
  for `--reaction`. Reaction defaults to all validated beam rows.
- `--seed N` selects a positive random seed.
- `--rng-state PATH` restores a saved full run-start state; it is mutually
  exclusive with `--seed`. Use the same inputs, configuration, and Geant4 build
  when replaying a run.

Default paths are relative to the working directory, so run from `build` as
shown. Each run recreates its configured outputs. Distinct reaction ROOT names
produce distinct truth, exit, metadata, and random-state filenames. Transmission
filenames retain the existing convention: `output/transmission/<DWBA-parent>.csv`.
A second run for the same state replaces that transmission file.

## Event selection and conventions

An S3 ring is hit when the **primary triton** deposits positive energy in it.
There is currently no additional electronics threshold. Several depositing
steps in the same ring count as one ring hit.

Ring numbers 16, 17, and 18 are excluded. Accept exactly one **enabled** ring:
one enabled ring plus any excluded-ring hits passes; two enabled rings fail.
The ROOT `S3` table and the S3 histograms contain only selected events. Event-level
ROOT/CSV records retain all reaction events for efficiency calculations.

- ROOT `ringID` is a zero-based index (0–23).
- CSV `triton_S3_ringID` is a one-based ring number (1–24).
- Laboratory polar angles are measured from global +z. Truth/exit CSV azimuths
  are normalized to 0–360 degrees. The S3 ROOT azimuth retains the signed value.
- `theta_cm_deg` is the sampled DWBA angle, applied to the triton about the axis
  opposite the incoming beam. `phi_cm_deg` uses the transverse basis constructed
  around that axis; it is not the global laboratory azimuth.

## Configuration and reconstruction

`include/mode.hh` groups beam-generation settings, reconstruction assumptions,
reaction inputs, output paths, and excluded ring numbers. Geometry dimensions
remain in the `DetectorConstruction` constructor and are recorded from its
actual getters in the run metadata.

`src/reconstruction.cxx` contains the detector-response smearing and CSDA
back-correction. The existing assumptions are retained:

- Smear the kinetic energy at the first depositing S3 step with Gaussian
  sigma 25 keV, clipping negative sampled energies to zero.
- Correct back through the silicon dead layer and half the LiF target using
  the ring-center angle, rather than the true event angle or reaction depth.
- Use the fixed reconstructed beam-center reference of 64.179880978 MeV.
  The original calibration provenance of this legacy value is not documented.
  It is distinct from the simulated event's actual beam energy.

Existing `edep_smeared_MeV` / `triton_S3_edep_smeared_MeV` column names are
retained for reader compatibility, but contain **smeared kinetic energy**.
`edep_MeV` and `hS3_total_edep` contain deposited energy.

## Outputs (schema version 2)

Preparation writes `beam.dat`, with columns:

```text
E_MeV x_mm y_mm z_mm dirx diry dirz depth_um
```

The values refer to the sampled reaction point. Direction components are
dimensionless, and depth is measured from the front of the LiF along z.
The preparation stream is opened once per run and checked on writes and close.

Reaction writes the existing ROOT file and three CSV products:

- `<stem>_truth_kinematics.csv`: generated vertex kinematics. It now appends
  `beamRowID`, `beam_E_MeV`, `beam_x_mm`, `beam_y_mm`, `beam_z_mm`, `beam_dirx`,
  `beam_diry`, `beam_dirz`, and `depth_um`. The zero-based beam row identifies the
  input record; each reaction run starts from row zero. These per-event values
  make depth analysis independent of later regeneration of the beam file.
- `<stem>_exit_kinematics.csv`: target-exit and selected S3/reconstruction values.
  It appends `nS3EnabledRingsHit` and `passesS3Selection` (0 or 1). Rejected events
  retain their rows with `nan` S3/reconstruction values.
- The transmission CSV keeps its three-column interface: `energy_MeV`,
  `theta_spec_deg`, `phi_spec_deg`. It requires both target exits plus the S3 cut.

The ROOT `reaction` table appends `hasMg26Exit`, `hasTritonExit`,
`nS3EnabledRingsHit`, and `passesS3Selection`. Missing exit kinematics are now
`NaN`, consistent with CSV; use the flags to distinguish missing measurements.
Existing ROOT table/histogram keys and pre-existing column order are preserved.
ROOT numeric IDs are captured at booking time in `src/analysis.cxx` and accessed
through the named schema in `include/analysis.hh`.

Both stages also write:

- `<stem>_metadata.json`: schema version, mode, run ID, requested/processed counts,
  current beam-generator settings, actual geometry, reconstruction settings,
  selection policy, Geant4/build information, input identities, output paths,
  and counts of events with zero/one/multiple enabled rings.
- `<stem>_random_start.rndm` and `<stem>_random_end.rndm`: complete CLHEP random
  states, including distribution caches. A seed alone cannot reproduce the
  reaction stage's starting state after a preceding preparation stage.

For preparation these filenames use the beam-file stem (e.g. `beam_metadata.json`).
Reaction metadata identifies its beam and DWBA files by absolute path, byte size,
and FNV-1a content fingerprint. This fingerprint is for provenance, not security.
The beam-generator settings in reaction metadata describe the current configured
preparation model; the actual reaction beam comes from the identified input file
and is recorded in each truth row.

Metadata starts with status `running` and becomes `complete` after outputs close
successfully. An interrupted/failed run may leave a `running` metadata record or
partial files. Reusing an output name replaces the preceding run's files.

## Analysis compatibility

`Plotting/simulation_io.py` provides the per-event beam reader used by the depth,
width-budget, SRIM, and TRIM analyses. New truth files supply their own beam
information. Legacy files fall back to `beam.dat`, with a warning if provenance
is unavailable; when a metadata identity exists, a mismatched beam file is
rejected. Existing legacy results are not rewritten.

## Regression checks

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DPython3_EXECUTABLE=/path/to/python
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The native checks cover malformed beam/DWBA data, output failures, four-momentum
conservation of generated primaries, and controlled S3 hits across repeated runs.
The Python workflow check uses only the standard library and exercises isolated
prepare/reaction runs, input/output diagnostics, and byte-for-byte CSV replay from
a saved random state. Python with `uproot` and `pandas` additionally enables ROOT
schema/selection checks and analysis provenance checks. CMake reports when those
optional checks are unavailable.

Test outputs are confined to `build/test-output` and temporary directories.
The suite does not regenerate the project's existing simulation datasets.
