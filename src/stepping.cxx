#include "stepping.hh"
#include "analysis.hh"
#include "output.hh"
#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "Randomize.hh"
#include "construction.hh"
#include "event.hh"
#include "mode.hh"
#include <cmath>

namespace {
G4bool IsTargetVolume(const G4String name) {
  return  name == "LiF_logic" || name == "C_logic";
}

void FillTargetExitKinematics(const G4Step *step, EventAction *eventAction) {
  auto *track = step->GetTrack();
  if (!track || track->GetParentID() != 0) {
    return;
  }
  auto *particle = track->GetDefinition();
  if (!particle) {
    return;
  }

  const G4int z = particle->GetAtomicNumber();
  const G4int a = particle->GetAtomicMass();

  const G4bool isMg26 = (z == 12 && a == 26);
  const G4bool isTriton = (z == 1 && a == 3);

  if (!isMg26 && !isTriton) {
    return;
  }

  auto *pre = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();

  if (!pre || !post) {
    return;
  }

  if (post->GetStepStatus() != fGeomBoundary) {
    return;
  }

  const auto *preVolume = pre->GetPhysicalVolume();
  const auto *postVolume = post->GetPhysicalVolume();

  if (!preVolume) {
    return;
  }

  const G4String preName = preVolume->GetLogicalVolume()->GetName();

  const G4String postName =
      postVolume ? postVolume->GetLogicalVolume()->GetName() : "";

  if (!IsTargetVolume(preName) || IsTargetVolume(postName)) {
    return;
  }

  const G4double energy = post->GetKineticEnergy();
  const G4ThreeVector direction = post->GetMomentumDirection();

  const G4double theta = direction.theta();
  G4double phi = direction.phi();

  if (phi < 0.0) {
    phi += 360.0 * deg;
  }



  auto *analysis = G4AnalysisManager::Instance();

  if (isMg26) {
    analysis->FillH2(Analysis::Get().recoilExit, theta / deg, energy / MeV);
    if (eventAction) {
      eventAction->StoreRecoil(energy, theta, phi);
    }
  } else if (isTriton) {
    analysis->FillH2(Analysis::Get().tritonExit, theta / deg, energy / MeV);
    if (eventAction) {
      eventAction->StoreLightParticle(energy, theta, phi);
    }
  }
}
} // namespace
SteppingAction::SteppingAction(EventAction *eventAction)
    : fEventAction(eventAction) {}

void SteppingAction::BeginRun() {
  // Event IDs restart at each run; do not retain the previous run's depth state.
  fCurrentEventID = -1;
  fWroteThisEvent = false;
  if (gSimulationMode != SimulationMode::Prepare) return;
  Output::Open(fBeamOut, gSimulationConfig.beamFile);
  fBeamOut << "# E_MeV x_mm y_mm z_mm dirx diry dirz depth_um\n";
  Output::Check(fBeamOut, gSimulationConfig.beamFile);
}

void SteppingAction::EndRun() {
  Output::Close(fBeamOut, gSimulationConfig.beamFile);
}

void SteppingAction::UserSteppingAction(const G4Step *step) {
  // check if we are in prepare mode
  if (gSimulationMode == SimulationMode::Reaction) {
    FillTargetExitKinematics(step, fEventAction);
    return;
  }
  if (gSimulationMode != SimulationMode::Prepare) {
    return;
  }
  G4Track *track = step->GetTrack();
  if (track->GetParentID() != 0) {
    return;
  }

  // check if it is 22Ne
  const G4ParticleDefinition *particle = track->GetDefinition();
  if (particle->GetAtomicNumber() != 10 || particle->GetAtomicMass() != 22) {
    return;
  }

  const G4Event *event = G4RunManager::GetRunManager()->GetCurrentEvent();
  if (!event) {
    return;
  }

  // Target:
  // |        LiF       | C |
  const G4int eventID = event->GetEventID();
  if (eventID != fCurrentEventID) {
    fCurrentEventID = eventID;
    fWroteThisEvent = false;

    const auto *detector = dynamic_cast<const DetectorConstruction *>(
        G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    if (!detector) {
      return;
    }

    const G4double lifThickness = detector->GetLiFThickness();
    const G4double lifFrontZ = detector->GetLiFFrontZ();

    // randomize reaction depth in LiF
    fReactionDepth = lifThickness * G4UniformRand();
    fReactionZ = lifFrontZ + fReactionDepth;
  }

  if (fWroteThisEvent) {
    return;
  }

  // check if the step has a begining or end, abort if not
  G4StepPoint *pre = step->GetPreStepPoint();
  G4StepPoint *post = step->GetPostStepPoint();
  if (!pre || !post) {
    return;
  }

  // check if the volumes are legal, abort if not
  const G4VPhysicalVolume *preVolume = pre->GetPhysicalVolume();
  const G4VPhysicalVolume *postVolume = post->GetPhysicalVolume();
  if (!preVolume || !postVolume) {
    return;
  }

  // we only care about the ones that cross LiF
  const G4String preName = preVolume->GetLogicalVolume()->GetName();
  const G4String postName = postVolume->GetLogicalVolume()->GetName();
  const G4bool involvesLiF =
      (preName == "LiF_logic" || postName == "LiF_logic");
  if (!involvesLiF) {
    return;
  }

  const G4ThreeVector prePos = pre->GetPosition();
  const G4ThreeVector postPos = post->GetPosition();

  const G4double z1 = prePos.z();
  const G4double z2 = postPos.z();

  // check if this step passed through the selected reaction depth
  // if not, keep waiting
  const G4bool crossesReactionZ = (z1 <= fReactionZ && fReactionZ <= z2) ||
                                  (z2 <= fReactionZ && fReactionZ <= z1);

  // avoid dividing by 0
  if (!crossesReactionZ || std::fabs(z2 - z1) == 0.0) {
    return;
  }

  // check where the reaction point lies
  G4double fraction = (fReactionZ - z1) / (z2 - z1);

  // exactly at the pre-step
  if (fraction < 0.0) {
    fraction = 0.0;
  }

  // exactly at post-step
  if (fraction > 1.0) {
    fraction = 1.0;
  }

  const G4ThreeVector pos = prePos + fraction * (postPos - prePos);
  const G4double preEnergy = pre->GetKineticEnergy();
  const G4double postEnergy = post->GetKineticEnergy();
  // beam energy at reaction point:
  const G4double energy = preEnergy + fraction * (postEnergy - preEnergy);

  G4ThreeVector dir =
      pre->GetMomentumDirection() +
      fraction * (post->GetMomentumDirection() - pre->GetMomentumDirection());

  // make dir a unit vectr
  if (dir.mag2() > 0.0) {
    dir = dir.unit();
  }

  fBeamOut << energy / MeV << " " << pos.x() / mm << " " << pos.y() / mm << " "
       << pos.z() / mm << " " << dir.x() << " " << dir.y() << " " << dir.z()
       << " " << fReactionDepth / um << "\n";

  Output::Check(fBeamOut, gSimulationConfig.beamFile);
  fWroteThisEvent = true;

  G4EventManager::GetEventManager()->AbortCurrentEvent();
}
