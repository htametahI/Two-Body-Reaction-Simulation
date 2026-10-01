#ifndef MODE_HH
#define MODE_HH

#include "G4String.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "globals.hh"
#include <vector>

// The simulation has a prepare mode and a reaction mode to save computing time
// Beam is sampled once and wirtten to beam.dat, reactions are simulated using the sampled beam file

// Beam properties: 
struct BeamConfig {
  G4double energy = 66.0 * MeV;
  G4double fractionalEnergySpread = 0.0017; // Gaussian 1sigma / mean energy.
  G4double spotDiameter = 0.66 * mm;
  G4double normalizedEmittance = 0.2 * CLHEP::pi * mm * mrad;
  G4double startZ = -2.0 * mm;
};

// For reconstructing the excitation energy spectrum
struct ReconstructionConfig {
  G4double s3EnergySigma = 25.0 * keV; // change depending on your S3 energy calibrations, this is meant to model the S3 energy resolution 
  G4double assumedBeamEnergyAtCenter = 64.179880978 * MeV; // calculate this assuming the beam reacts at the center of your target
};

enum class SimulationMode { Prepare, Reaction };

// simulation input and outputs
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
  // exlucded ring numbers (change if they are good in your experiment)
  std::vector<G4int> excludedS3RingNumbers = {16, 17, 18};
};

extern SimulationMode gSimulationMode;
extern SimulationConfig gSimulationConfig;
bool IsEnabledS3RingIndex(G4int ringIndex);
void ValidateSimulationConfig();

#endif
