#ifndef ANALYSIS_HH
#define ANALYSIS_HH

// IDs are captured from Geant4 at booking time. Writers never rely on the
// creation order of histograms, ntuples, or columns.
namespace Analysis {
struct ParticleColumns { int energy, theta, phi; };
struct ReactionColumns {
  int id, eventID;
  ParticleColumns recoil, triton;
  int hasRecoilExit, hasTritonExit, enabledRingCount, passesS3Selection;
};
struct S3Columns {
  int id, eventID, ringIndex, depositedEnergy, smearedKineticEnergy;
  int kineticEnergy, theta, phi;
};
struct Schema {
  int recoilExit, tritonExit, recoilTruth, tritonTruth;
  int s3Energy, s3EnergyByRing, s3TotalDeposit;
  ReactionColumns reaction;
  S3Columns s3;
};
const Schema &Get();
void Book();
}
#endif
