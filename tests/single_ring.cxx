#include "construction.hh"
#include "event.hh"
#include "mode.hh"
#include "physics.hh"
#include "run.hh"
#include "G4Event.hh"
#include "G4IonTable.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"
#include <iostream>
#include <fstream>
#include <map>
#include <vector>

// Each entry lists zero-based rings into which test primaries are injected.
const std::vector<std::vector<int>> cases = {
  {}, {1}, {1, 5}, {1, 1}, {15}, {1, 15},
  {1, 5, 15}, {23}, {15, 16}, {}, {0}, {1, 16, 17}
};
std::vector<std::map<int, int>> depositingSteps(cases.size());

class TestPrimaries : public G4VUserPrimaryGeneratorAction {
public:
  explicit TestPrimaries(const DetectorConstruction *detector)
      : fDetector(detector), fGun(1) {}

  void GeneratePrimaries(G4Event *event) override {
    fGun.SetParticleDefinition(
        G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(1, 3, 0.0));
    fGun.SetParticleCharge(eplus);
    fGun.SetParticleEnergy(1.0 * MeV);
    fGun.SetParticleMomentumDirection(G4ThreeVector(0., 0., -1.));
    const auto z = fDetector->GetTargetZOffset() +
        0.5 * (fDetector->GetLiFThickness() + fDetector->GetCThickness()) -
        fDetector->GetS3DistanceFromTargetCenter();
    const auto width = (fDetector->GetS3OuterRadius() -
        fDetector->GetS3InnerRadius()) / fDetector->GetS3RingCount();
    const auto &rings = cases.at(event->GetEventID());
    if (rings.empty()) {
      fGun.SetParticlePosition(G4ThreeVector(0., 0., z));
      fGun.GeneratePrimaryVertex(event);
    }
    for (int ring : rings) {
      const auto radius = fDetector->GetS3InnerRadius() + (ring + 0.5) * width;
      fGun.SetParticlePosition(G4ThreeVector(radius, 0., z));
      fGun.GeneratePrimaryVertex(event);
    }
  }
private:
  const DetectorConstruction *fDetector;
  G4ParticleGun fGun;
};

class TestEvent : public EventAction {
public:
  using EventAction::EventAction;
  void BeginOfEventAction(const G4Event *event) override {
    EventAction::BeginOfEventAction(event);
    // Supply exit-kinematics fixtures to isolate the S3 transmission gate.
    if (event->GetEventID() != 0 && event->GetEventID() != 9) {
      if (event->GetEventID() != 10) StoreRecoil(40.0 * MeV, 2.0 * deg, 0.0);
      StoreLightParticle(1.0 * MeV, 160.0 * deg, 0.0);
    }
  }
};

class ObserveSteps : public G4UserSteppingAction {
public:
  void UserSteppingAction(const G4Step *step) override {
    const auto *track = step->GetTrack();
    const auto *particle = track->GetDefinition();
    if (track->GetParentID() != 0 || particle->GetAtomicNumber() != 1 ||
        particle->GetAtomicMass() != 3 || step->GetTotalEnergyDeposit() <= 0.) {
      return;
    }
    const auto *pre = step->GetPreStepPoint();
    const auto *volume = pre->GetPhysicalVolume();
    if (!volume || volume->GetLogicalVolume()->GetName().find("s3Ring_logic_") != 0) {
      return;
    }
    const int eventID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    ++depositingSteps.at(eventID)[pre->GetTouchableHandle()->GetCopyNumber()];
  }
};

int main() {
  CLHEP::HepRandom::setTheSeed(982451653);
  gSimulationMode = SimulationMode::Reaction;
  gSimulationConfig.outputFile = "check.root";
  gSimulationConfig.exitKinematicsFile = "check_exit.csv";
  gSimulationConfig.transmissionFile = "check_transmission.csv";
  G4RunManager manager;
  auto *detector = new DetectorConstruction();
  manager.SetUserInitialization(detector);
  manager.SetUserInitialization(new PhysicsList());
  manager.SetUserAction(new TestPrimaries(detector));
  auto *run = new RunAction();
  auto *event = new TestEvent(run);
  run->SetActions(nullptr, event, nullptr);
  manager.SetUserAction(run);
  manager.SetUserAction(event);
  manager.SetUserAction(new ObserveSteps());
  manager.Initialize();
  manager.BeamOn(cases.size());
  // A second run in the same process checks output and event-state reset.
  manager.BeamOn(cases.size());
  std::ofstream observed("observed.txt");
  for (std::size_t event = 0; event < depositingSteps.size(); ++event) {
    observed << "OBSERVED " << event;
    for (const auto &[ring, steps] : depositingSteps[event]) {
      observed << " " << ring << ":" << steps;
    }
    observed << "\n";
  }
}
