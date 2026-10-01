#include "analysis.hh"
#include "G4AnalysisManager.hh"

namespace Analysis {
namespace {
// Matches Geant4's per-thread analysis manager; the executable currently uses
// a sequential run manager. This does not make the text writers multithreaded.
thread_local Schema schema{};
}
const Schema &Get() { return schema; }

void Book() {
  auto *man = G4AnalysisManager::Instance();
  man->SetDefaultFileType("root");
  man->SetVerboseLevel(1);
  schema.recoilExit = man->CreateH2("hMg26_exit_E_vs_theta",
      "26Mg after target exit;theta_lab [deg];E [MeV]", 180, 0., 180., 400, 0., 200.);
  schema.tritonExit = man->CreateH2("hTriton_exit_E_vs_theta",
      "triton after target exit;theta_lab [deg];E [MeV]", 180, 0., 180., 400, 0., 50.);
  // Keep existing ROOT keys for readers; titles describe the actual observable.
  schema.s3Energy = man->CreateH1("hS3_edep",
      "S3 smeared triton kinetic energy, single enabled ring;E [MeV];Counts", 400, 0., 50.);
  schema.s3TotalDeposit = man->CreateH1("hS3_total_edep",
      "S3 deposited energy, single enabled ring;E_dep [MeV];Counts", 400, 0., 50.);
  schema.s3EnergyByRing = man->CreateH2("hS3_edep_vs_ring",
      "S3 smeared triton kinetic energy;Ring index (0-based);E [MeV]", 24, -0.5, 23.5, 400, 0., 50.);
  schema.recoilTruth = man->CreateH2("hMg26_truth_E_vs_theta",
      "26Mg at reaction vertex;theta_lab [deg];E [MeV]", 180, 0., 180., 400, 0., 200.);
  schema.tritonTruth = man->CreateH2("hTriton_truth_E_vs_theta",
      "triton at reaction vertex;theta_lab [deg];E [MeV]", 180, 0., 180., 400, 0., 50.);

  auto &r = schema.reaction;
  r.id = man->CreateNtuple("reaction", "Target-exit kinematics and S3 selection");
  r.eventID = man->CreateNtupleIColumn(r.id, "eventID");
  r.recoil.energy = man->CreateNtupleDColumn(r.id, "Mg26_E_MeV");
  r.recoil.theta = man->CreateNtupleDColumn(r.id, "Mg26_theta_deg");
  r.recoil.phi = man->CreateNtupleDColumn(r.id, "Mg26_phi_deg");
  r.triton.energy = man->CreateNtupleDColumn(r.id, "triton_E_MeV");
  r.triton.theta = man->CreateNtupleDColumn(r.id, "triton_theta_deg");
  r.triton.phi = man->CreateNtupleDColumn(r.id, "triton_phi_deg");
  r.hasRecoilExit = man->CreateNtupleIColumn(r.id, "hasMg26Exit");
  r.hasTritonExit = man->CreateNtupleIColumn(r.id, "hasTritonExit");
  r.enabledRingCount = man->CreateNtupleIColumn(r.id, "nS3EnabledRingsHit");
  r.passesS3Selection = man->CreateNtupleIColumn(r.id, "passesS3Selection");
  man->FinishNtuple(r.id);

  auto &s = schema.s3;
  s.id = man->CreateNtuple("S3", "Single-enabled-ring S3 hits");
  s.eventID = man->CreateNtupleIColumn(s.id, "eventID");
  s.ringIndex = man->CreateNtupleIColumn(s.id, "ringID");
  s.depositedEnergy = man->CreateNtupleDColumn(s.id, "edep_MeV");
  // Legacy branch name retained for existing ROOT readers.
  s.smearedKineticEnergy = man->CreateNtupleDColumn(s.id, "edep_smeared_MeV");
  s.kineticEnergy = man->CreateNtupleDColumn(s.id, "ekin_MeV");
  s.theta = man->CreateNtupleDColumn(s.id, "theta_deg");
  s.phi = man->CreateNtupleDColumn(s.id, "phi_deg");
  man->FinishNtuple(s.id);
}
}
