#ifndef EVENT_HH
#define EVENT_HH

#include "G4UserEventAction.hh"
#include "globals.hh"
#include <fstream>

class G4Event;
class RunAction;

class EventAction : public G4UserEventAction {
public:
  explicit EventAction(RunAction *runAction = nullptr) : fRunAction(runAction) {}
  void BeginRun();
  void EndRun();
  void BeginOfEventAction(const G4Event *) override;
  void EndOfEventAction(const G4Event *) override;

  void StoreRecoil(G4double energy, G4double theta, G4double phi);
  void StoreLightParticle(G4double energy, G4double theta, G4double phi);
  void StoreS3Multiplicity(G4int enabledRingCount);
  void StoreS3RingHit(G4int ringNumber, G4double smearedKineticEnergy,
                     G4double tritonEnergyAtCenter, G4double beamEnergyAtCenter);

private:
  RunAction *fRunAction = nullptr; // Geant4 owns the actions.
  std::ofstream fExitOut, fTransmissionOut;
  G4bool fHasTritonExit = false, fGotMg = false, fGotS3Hit = false;
  G4int fS3EnabledRingsHit = 0;
  G4int fS3RingNumber = -1; // 1-based; -1 means no selected ring.
  G4double fRecoilEnergy = 0., fRecoilTheta = 0., fRecoilPhi = 0.;
  G4double fLightEnergy = 0., fLightTheta = 0., fLightPhi = 0.;
  G4double fS3SmearedKineticEnergy = 0.;
  G4double fTritonEnergyAtCenter = 0., fBeamEnergyAtCenter = 0.;
};
#endif
