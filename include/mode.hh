#ifndef MODE_HH
#define MODE_HH

#include "G4String.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "globals.hh"
#include <vector>

// Values here are run inputs. The simulated beam energy and the energy assumed
// during reconstruction are deliberately separate quantities.
struct BeamConfig {
  G4double energy = 66.0 * MeV;
  G4double fractionalEnergySpread = 0.0017; // Gaussian sigma / mean energy.
  G4double spotDiameter = 0.66 * mm;
  G4double normalizedEmittance = 0.2 * CLHEP::pi * mm * mrad;
  G4double startZ = -2.0 * mm;
};

struct ReconstructionConfig {
  G4double s3EnergySigma = 25.0 * keV;
  // Retained reference value from the existing reconstruction. Its original
  // calibration provenance is not recorded; do not substitute event truth here.
  G4double assumedBeamEnergyAtCenter = 64.179880978 * MeV;
};

enum class SimulationMode { Prepare, Reaction };

struct SimulationConfig {
  G4String beamFile = "../output/beam.dat";
  G4String dwbaFile = "../input/11500_1minus/fort.202";
  G4String outputFile = "../output/reaction.root";
  G4String truthKinematicsFile = "../output/reaction_truth_kinematics.csv";
  G4String exitKinematicsFile = "../output/reaction_exit_kinematics.csv";
  G4String transmissionFile = "../output/transmission/11500_1minus.csv";
  G4double excitationEnergy = 11.5 * MeV;
  BeamConfig beam;
  ReconstructionConfig reconstruction;
  // exlucded ring numbers
  std::vector<G4int> excludedS3RingNumbers = {16, 17, 18};
};

extern SimulationMode gSimulationMode;
extern SimulationConfig gSimulationConfig;
bool IsEnabledS3RingIndex(G4int ringIndex);
void ValidateSimulationConfig();

#endif
