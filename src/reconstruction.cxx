#include "reconstruction.hh"
#include "construction.hh"
#include "mode.hh"
#include "G4EmCalculator.hh"
#include "G4NistManager.hh"
#include "G4IonTable.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
// calculate lab angle at the center of a particular ring 
G4double S3RingCenterTheta(const DetectorConstruction &detector,
                           G4int ringIndex) {
  const G4double ringWidth =
      (detector.GetS3OuterRadius() - detector.GetS3InnerRadius()) /
      detector.GetS3RingCount();

  const G4double rCenter =
      detector.GetS3InnerRadius() + (ringIndex + 0.5) * ringWidth;

  // S3 is upstream of target, so lab theta is backward:
  // theta = pi - atan(r / distance)
  return CLHEP::pi -
         std::atan2(rCenter, detector.GetS3DistanceFromTargetCenter());
}

// obtain energy loss values from G4 CSDA tables by inverting the range table
G4double EnergyFromCSDARange(G4EmCalculator &calc, G4double targetRange,
                             const G4Material *material,
                             const G4ParticleDefinition *particle,
                             G4double highguess) {
  if (targetRange <= 0.0) {
    return 0.0;
  }
  G4double low = 0.0;
  G4double high = std::max(highguess, 1.0 * keV);
  while (calc.GetCSDARange(high, particle, material) < targetRange) {
    high *= 2.0;
    if (high > 2.0 * GeV) {
      throw std::runtime_error("CSDA range inversion exceeded 2 GeV");
    }
  }
  for (G4int i = 0; i < 80; ++i) {
    const G4double mid = 0.5 * (low + high);
    const G4double midRange = calc.GetCSDARange(mid, particle, material);

    if (midRange < targetRange) {
      low = mid;
    } else {
      high = mid;
    }
  }

  return 0.5 * (low + high);
}

G4double AbsCos(G4double theta) {
  const G4double c = std::abs(std::cos(theta));
  return c > 1.0e-6 ? c : 1.0e-6;
}


// backwards energy correction:
G4double BackCorrectLayer(G4EmCalculator &calc, G4double eAfter,
                          G4double pathLength, const G4Material *material,
                          const G4ParticleDefinition *particle) {
  if (eAfter <= 0.0) {
    return 0.0;
  }
  // find the range after the triton has traversed through a certain material
  const G4double rangeAfter = calc.GetCSDARange(eAfter, particle, material);
  // find the energy corresponding to that range
  return EnergyFromCSDARange(calc, rangeAfter + pathLength, material, particle,
                             eAfter);
}

} // namespace

S3Reconstruction ReconstructS3Hit(G4double kineticEnergy, G4int ringIndex,
    const DetectorConstruction &detector, const ReconstructionConfig &config) {
  if (ringIndex < 0 || ringIndex >= detector.GetS3RingCount() ||
      !std::isfinite(kineticEnergy) || kineticEnergy < 0.0) {
    throw std::runtime_error("Invalid S3 reconstruction input");
  }
  auto smeared = CLHEP::RandGauss::shoot(kineticEnergy, config.s3EnergySigma);
  smeared = std::max(0.0, smeared);
  G4EmCalculator calc;
  auto *nist = G4NistManager::Instance();
  auto *si = nist->FindOrBuildMaterial("G4_Si");
  auto *lif = nist->FindOrBuildMaterial("G4_LITHIUM_FLUORIDE");
  auto *triton = G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(1, 3, 0.0);
  const auto cosTheta = AbsCos(S3RingCenterTheta(detector, ringIndex));
  const auto siPath = detector.GetSiDeadLayerThickness() / cosTheta;
  const auto lifPath = 0.5 * detector.GetLiFThickness() / cosTheta;
  const auto beforeDeadLayer = BackCorrectLayer(calc, smeared, siPath, si, triton);
  const auto atCenter = BackCorrectLayer(calc, beforeDeadLayer, lifPath, lif, triton);
  return {smeared, atCenter, config.assumedBeamEnergyAtCenter};
}
