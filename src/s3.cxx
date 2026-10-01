#include "s3.hh"
#include "analysis.hh"
#include "construction.hh"
#include "event.hh"
#include "mode.hh"
#include "reconstruction.hh"
#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4ParticleDefinition.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include <algorithm>
#include <stdexcept>

S3::S3(const G4String &name, G4int ringCount)
    : G4VSensitiveDetector(name), fRingCount(ringCount), fRingHits(ringCount) {}

void S3::Initialize(G4HCofThisEvent *) {
  for (auto &hit : fRingHits) hit = RingHit{};
}

G4bool S3::ProcessHits(G4Step *step, G4TouchableHistory *) {
  const auto *track = step->GetTrack();
  if (!track || track->GetParentID() != 0) return false;
  const auto *particle = track->GetDefinition();
  if (particle->GetAtomicNumber() != 1 || particle->GetAtomicMass() != 3) return false;
  const auto edep = step->GetTotalEnergyDeposit();
  const auto *pre = step->GetPreStepPoint();
  if (edep <= 0.0 || !pre) return false;
  const auto ringIndex = pre->GetTouchableHandle()->GetCopyNumber();
  if (ringIndex < 0 || ringIndex >= fRingCount || !IsEnabledS3RingIndex(ringIndex)) return false;

  // Multiple depositing steps in one enabled ring still count as one ring hit.
  auto &hit = fRingHits[ringIndex];
  hit.edep += edep;
  if (!hit.hit) {
    hit.hit = true;
    hit.ekin = pre->GetKineticEnergy();
    hit.theta = pre->GetMomentumDirection().theta();
    hit.phi = pre->GetMomentumDirection().phi();
  }
  return true;
}

void S3::EndOfEvent(G4HCofThisEvent *) {
  if (gSimulationMode != SimulationMode::Reaction) return;
  auto *manager = G4RunManager::GetRunManager();
  const auto *event = manager->GetCurrentEvent();
  auto *eventAction = const_cast<EventAction *>(
      dynamic_cast<const EventAction *>(manager->GetUserEventAction()));
  if (!event || !eventAction) throw std::logic_error("S3 requires an event and EventAction");

  const auto hitRingCount = static_cast<G4int>(std::count_if(
      fRingHits.begin(), fRingHits.end(), [](const RingHit &hit) { return hit.hit; }));
  // Store multiplicity before rejecting: zero and multiple hits must remain
  // distinguishable in event outputs and run totals. Disabled rings do not count.
  eventAction->StoreS3Multiplicity(hitRingCount);
  if (hitRingCount != 1) return;

  const auto *detector = dynamic_cast<const DetectorConstruction *>(manager->GetUserDetectorConstruction());
  if (!detector) throw std::logic_error("S3 requires DetectorConstruction");
  const auto it = std::find_if(fRingHits.begin(), fRingHits.end(),
                             [](const RingHit &hit) { return hit.hit; });
  const auto ringIndex = static_cast<G4int>(std::distance(fRingHits.begin(), it));
  const auto &hit = *it;
  const auto reco = ReconstructS3Hit(hit.ekin, ringIndex, *detector,
                                     gSimulationConfig.reconstruction);
  // Preserve the external conventions: ROOT uses indices; CSV uses ring numbers.
  eventAction->StoreS3RingHit(ringIndex + 1, reco.smearedKineticEnergy,
                             reco.tritonEnergyAtCenter, reco.assumedBeamEnergyAtCenter);

  auto *man = G4AnalysisManager::Instance();
  const auto &schema = Analysis::Get();
  man->FillH1(schema.s3Energy, reco.smearedKineticEnergy / MeV);
  man->FillH2(schema.s3EnergyByRing, ringIndex, reco.smearedKineticEnergy / MeV);
  man->FillH1(schema.s3TotalDeposit, hit.edep / MeV);
  const auto &s = schema.s3;
  man->FillNtupleIColumn(s.id, s.eventID, event->GetEventID());
  man->FillNtupleIColumn(s.id, s.ringIndex, ringIndex);
  man->FillNtupleDColumn(s.id, s.depositedEnergy, hit.edep / MeV);
  man->FillNtupleDColumn(s.id, s.smearedKineticEnergy, reco.smearedKineticEnergy / MeV);
  man->FillNtupleDColumn(s.id, s.kineticEnergy, hit.ekin / MeV);
  man->FillNtupleDColumn(s.id, s.theta, hit.theta / deg);
  man->FillNtupleDColumn(s.id, s.phi, hit.phi / deg);
  man->AddNtupleRow(s.id);
}
