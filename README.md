# Two-Body Reaction Simulation

![Simulation geometry and particle trajectories](<Screenshot 2026-10-01 at 10.54.30 AM.png>)

## Table of Contents

- [System Requirements](#system-requirements)
- [Download and Build](#download-and-build)
- [Basics](#basics)
- [Running the Simulation](#running-the-simulation)
- [Output Files](#output-files)
- [Notes](#notes)

## System Requirements

Simulations are built with:

- macOS Tahoe 26.2 (M1 Chip)
- `ROOT` 6.36.06
- `GEANT4` 11.3.2
- `CMake` 4.2.1
- Apple `clang` version 17.0.0 (clang-1700.6.3.2)

## Download and Build

```bash
git clone https://github.com/htametahI/Two-Body-Reaction-Simulation.git
cd Two-Body-Reaction-Simulation
cmake -S . -B build
cmake --build build --parallel
cd build
make
```

If CMake cannot find Geant4, provide the directory containing `Geant4Config.cmake` for your installation:

```bash
cmake -S . -B build -DGeant4_DIR=/path/to/geant4/lib/cmake/Geant4
```

Run all simulation commands from `build/`. Default paths are relative
to the working directory: inputs are in `../input/` and outputs in `../output/`.
Output directories are created automatically.

After making changes to the simulation, call `make` in the `build` directory to recompile.

## Basics

This is a two-body reaction and detector response simulation. In order to save computing time, the beam preparation and reaction generation are separate, such that the beam only needs to be sampled once.

1. **Prepare the beam.** Sample the incident beam energy, position, and direction,
   transport it into the target, and save its state at a sampled reaction depth
   in `beam.dat`.
2. **Generate the reaction.** Read one saved beam row per reaction event. If DWBA calculations are available (`fort.202` file), sample the center-of-mass angle from the DWBA distribution, including the `sin(theta)` solid-angle weight, and sample a uniform azimuth. Calculate relativistic two-body kinematics for the requested recoil excitation energy and transform the products to the laboratory frame.
3. **Transport and detect.** Geant4 transports the products through the target
   and detector. The energy and angle of the recoils and ejectiles are recorded at the reaction vertices and the target exit.
4. **Select and reconstruct.** Accept events with exactly one enabled S3 ring
   hit. Multi-hit option is available. Apply detector-energy smearing and back-correct the triton energy through
   the silicon dead layer and half the LiF target thickness using CSDA ranges.

The DWBA table sets the angular sampling shape; generated event counts are not
an absolute experimental yield. Reaction species are currently hard-coded, rather than selected through command-line arguments.

## Running the Simulation

The simulation can be run in several modes as listed below, `[ ]` indicates optional arguments.

| Mode | Command | Behavior |
| --- | --- | --- |
| Default | `./sim` | Prepare 1,000 beam events, then simulate reactions using the saved rows |
| Preparation | `./sim --prepare N` | Generate a reusable beam file |
| Reaction | `./sim --reaction DWBA Ex_MeV OUTPUT.root [N]` | Read an existing beam file and simulate one excitation state |
| Interactive | `./sim --vis` | Prepare the beam, display reactions, and open the Geant4 UI |
| Macro | `./sim MACRO` | Execute a Geant4 macro in preparation mode |

## Output Files

| File in `output/` | Contents |
| --- | --- |
| `beam.dat` | Beam state at each sampled reaction point |
| `<state>.root` | Reaction/S3 event tables and energy-angle histograms |
| `<state>_truth_kinematics.csv` | Generated vertex kinematics, CM angles, and the beam state used for each event |
| `<state>_exit_kinematics.csv` | Target-exit kinematics and selected S3/reconstructed energies. |
| `transmission/<state>.csv` | Recoil energy and angles at target exit in EMMA spectrometer coordinates for transmission efficiency calculations. |

The contents of each `.csv` file are as follows. Note that the current labels are set up for a particular experiment and yours might differ. ²⁶Mg → recoils; triton → ejectiles.

### `<state>_truth_kinematics.csv`

One row per successfully generated reaction, **before** particle transport. This
file includes events regardless of whether they later pass the S3 selection.

| Column | Unit | Contents |
| --- | --- | --- |
| `eventID` | — | Event number within the current run. |
| `Ex_MeV` | MeV | Requested excitation energy of ²⁶Mg. |
| `theta_cm_deg` | degrees | Sampled triton CM polar angle. |
| `phi_cm_deg` | degrees | Sampled triton CM azimuthal angle. |
| `Mg26_E_MeV` | MeV | ²⁶Mg kinetic energy at the reaction vertex. |
| `Mg26_theta_deg` | degrees | ²⁶Mg laboratory polar angle at the vertex. |
| `Mg26_phi_deg` | degrees | ²⁶Mg laboratory azimuthal angle at the vertex. |
| `triton_E_MeV` | MeV | Triton kinetic energy at the reaction vertex. |
| `triton_theta_deg` | degrees | Triton laboratory polar angle at the vertex, measured from global +z. |
| `triton_phi_deg` | degrees | Triton laboratory azimuthal angle at the vertex. |
| `beamRowID` | — | Index of the input beam row used for this reaction. |
| `beam_E_MeV` | MeV | Saved incident beam kinetic energy at the reaction point. |
| `beam_x_mm` | mm | Reaction-point x coordinate from the beam file. |
| `beam_y_mm` | mm | Reaction-point y coordinate from the beam file. |
| `beam_z_mm` | mm | Reaction-point z coordinate from the beam file. |
| `beam_dirx` | dimensionless | Saved beam-direction x component; the generator normalizes the direction before use. |
| `beam_diry` | dimensionless | Saved beam-direction y component. |
| `beam_dirz` | dimensionless | Saved beam-direction z component. |
| `depth_um` | µm | Saved reaction depth from the LiF front surface along global z. |

### `<state>_exit_kinematics.csv`

One row per reaction event recorded at the end of the event, including events
that fail the S3 selection. Match `eventID` with the truth file from the same run
to compare vertex and exit kinematics.

| Column | Unit | Contents |
| --- | --- | --- |
| `eventID` | — | Event number within the current run. |
| `Ex_MeV` | MeV | Requested excitation energy of ²⁶Mg. |
| `hasMg26Exit` | 0 or 1 | Whether a ²⁶Mg target exit was recorded. |
| `Mg26_E_MeV` | MeV | ²⁶Mg kinetic energy at the recorded target exit. |
| `Mg26_theta_deg` | degrees | ²⁶Mg laboratory polar angle at exit. |
| `Mg26_phi_deg` | degrees | ²⁶Mg laboratory azimuthal angle. |
| `hasTritonExit` | 0 or 1 | Whether a triton target exit was recorded. |
| `triton_E_MeV` | MeV | Triton kinetic energy at the recorded target exit. |
| `triton_theta_deg` | degrees | Triton laboratory polar angle at exit. |
| `triton_phi_deg` | degrees | Triton laboratory azimuthal angle at exit. |
| `triton_S3_ringID` | — | S3 ring number for events that have an S3 hit, using one-based numbering (1–24). |
| `triton_S3_edep_smeared_MeV` | MeV | Smeared triton kinetic energy (change this value for your experiment) at its first depositing step in the selected ring. |
| `triton_E_reco_center_MeV` | MeV | Triton energy back-corrected through the silicon dead layer and half the LiF thickness using the ring-center angle. |
| `beam_E_center_MeV` | MeV | Fixed assumed beam energy at the target center used for reconstruction, not the event's true beam energy. |
| `nS3EnabledRingsHit` | count | Number of enabled rings with positive primary-triton energy deposition. |
| `passesS3Selection` | 0 or 1 | 1 if exactly one enabled ring was hit; otherwise 0. |

If a particle has no recorded target exit, its exit energy and angles are `nan`.
If an event does not hit the S3, the four columns from `triton_S3_ringID`
through `beam_E_center_MeV` are `nan`.

### `transmission/<DWBA-state>.csv`

One row per event with both target exits recorded and exactly one enabled S3
ring hit. `<DWBA-state>` is the input table's parent-folder name, for example
`11500_1minus` for `input/11500_1minus/fort.202`.

| Column | Unit | Contents |
| --- | --- | --- |
| `energy_MeV` | MeV | Selected ²⁶Mg recoil kinetic energy at target exit. |
| `theta_spec_deg` | degrees | Projected recoil angle in the x–z plane: `atan(tan(theta) * cos(phi))`. |
| `phi_spec_deg` | degrees | Projected recoil angle in the y–z plane: `atan(tan(theta) * sin(phi))`. |

Here `theta` and `phi` are the recoil's laboratory polar and azimuthal angles
at exit. The code evaluates trigonometric functions in radians and writes the
results in degrees. The output angles are in the EMMA spectrometer coordinate system.

## Notes

- Right now ring 15, 16 and 17 are disabled. Modify the code for your experiment if needed.
- The beam energy at the target center is pre-computed with SRIM and is currently set to 64.179880978 MeV. Change this for your experiment.
- There are `.rndm` and `.json` files that are generated in the `output` directory. They record the state of the simulations.
