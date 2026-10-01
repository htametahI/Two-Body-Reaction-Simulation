#ifndef RECONSTRUCTION_HH
#define RECONSTRUCTION_HH
#include "globals.hh"

class DetectorConstruction;
struct ReconstructionConfig;
struct S3Reconstruction {
  G4double smearedKineticEnergy;
  G4double tritonEnergyAtCenter;
  G4double assumedBeamEnergyAtCenter;
};

// Input is the kinetic energy at the first depositing S3 step, after the dead
// layer. Reconstruction uses the ring-center angle and half the LiF thickness,
// not the true event position or direction.
S3Reconstruction ReconstructS3Hit(G4double kineticEnergy, G4int ringIndex,
    const DetectorConstruction &detector, const ReconstructionConfig &config);
#endif
