#ifndef RUN_HH
#define RUN_HH

#include "G4UserRunAction.hh"
#include <array>
#include <string>

class PrimaryGenerator;
class EventAction;
class SteppingAction;
class G4Run;

class RunAction : public G4UserRunAction {
public:
  RunAction();
  // Non-owning links let run boundaries control every text stream explicitly.
  void SetActions(PrimaryGenerator *generator, EventAction *event, SteppingAction *stepping);
  void BeginOfRunAction(const G4Run *run) override;
  void EndOfRunAction(const G4Run *run) override;
  void RecordS3Multiplicity(int enabledRingCount);

private:
  void WriteMetadata(const G4Run *run, const std::string &status);
  PrimaryGenerator *fGenerator = nullptr;
  EventAction *fEvent = nullptr;
  SteppingAction *fStepping = nullptr;
  std::array<unsigned long long, 3> fS3Counts{}; // zero, one, multiple enabled rings
  std::string fMetadataPath, fMetadataPrefix, fRandomEndPath;
};
#endif
