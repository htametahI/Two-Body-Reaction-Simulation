#ifndef INPUT_HH
#define INPUT_HH

#include "globals.hh"
#include <string>
#include <vector>

// Values retain the beam-file units: MeV, mm, dimensionless direction, um.
// Conversion to Geant4 units occurs when generating the reaction.
struct BeamState {
  G4double energy, x, y, z, ux, uy, uz, depth;
};

struct DwbaDistribution {
  std::vector<G4double> theta; // Geant4 angular units (radians).
  std::vector<G4double> cdf;
};

std::vector<BeamState> ReadBeamFile(const std::string &path);
DwbaDistribution ReadDwbaFile(const std::string &path);

#endif
