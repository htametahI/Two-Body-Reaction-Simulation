#ifndef GENERATOR_HH
#define GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "input.hh"
#include <cstddef>
#include <fstream>
#include <vector>

class G4Event;
class G4ParticleGun;

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
public:
  PrimaryGenerator();
  ~PrimaryGenerator() override;
  void BeginRun(G4int requestedEvents = -1);
  void EndRun();
  void GeneratePrimaries(G4Event *event) override;

private:
  void GenerateBeam(G4Event *event);
  void GenerateReaction(G4Event *event);
  G4double SampleDWBA() const;
  G4ParticleGun *fParticleGun = nullptr;
  std::vector<BeamState> fBeamStates;
  std::size_t fNextBeamState = 0;
  DwbaDistribution fDwba;
  std::ofstream fTruthOut;
};
#endif
