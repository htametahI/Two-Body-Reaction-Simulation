#include "mode.hh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

SimulationMode gSimulationMode = SimulationMode::Prepare;
SimulationConfig gSimulationConfig;

bool IsEnabledS3RingIndex(G4int ringIndex) {
  const auto &excluded = gSimulationConfig.excludedS3RingNumbers;
  return std::find(excluded.begin(), excluded.end(), ringIndex + 1) == excluded.end();
}

void ValidateSimulationConfig() {
  const auto &c = gSimulationConfig;
  const auto &b = c.beam;
  const auto &r = c.reconstruction;
  if (!std::isfinite(c.excitationEnergy) || c.excitationEnergy < 0.0 ||
      !std::isfinite(b.energy) || b.energy <= 0.0 ||
      !std::isfinite(b.fractionalEnergySpread) || b.fractionalEnergySpread < 0.0 ||
      !std::isfinite(b.spotDiameter) || b.spotDiameter < 0.0 ||
      !std::isfinite(b.normalizedEmittance) || b.normalizedEmittance < 0.0 ||
      !std::isfinite(b.startZ) ||
      !std::isfinite(r.s3EnergySigma) || r.s3EnergySigma < 0.0 ||
      !std::isfinite(r.assumedBeamEnergyAtCenter) || r.assumedBeamEnergyAtCenter <= 0.0) {
    throw std::runtime_error("Invalid beam, excitation, or reconstruction configuration");
  }
}
