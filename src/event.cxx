#include "event.hh"
#include "analysis.hh"
#include "mode.hh"
#include "output.hh"
#include "run.hh"
#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <limits>
#include <stdexcept>

void EventAction::BeginRun() {
  if (gSimulationMode != SimulationMode::Reaction) return;
  Output::Open(fExitOut, gSimulationConfig.exitKinematicsFile);
  // Preserve existing columns, including the legacy "edep_smeared" name.
  // It stores smeared kinetic energy. New selection fields are appended.
  fExitOut << "eventID,Ex_MeV,"
           << "hasMg26Exit,Mg26_E_MeV,Mg26_theta_deg,Mg26_phi_deg,"
           << "hasTritonExit,triton_E_MeV,triton_theta_deg,triton_phi_deg,"
           << "triton_S3_ringID,triton_S3_edep_smeared_MeV,"
           << "triton_E_reco_center_MeV,beam_E_center_MeV,"
           << "nS3EnabledRingsHit,passesS3Selection\n";
  Output::Check(fExitOut, gSimulationConfig.exitKinematicsFile);
  Output::Open(fTransmissionOut, gSimulationConfig.transmissionFile);
  fTransmissionOut << "energy_MeV,theta_spec_deg,phi_spec_deg\n";
  Output::Check(fTransmissionOut, gSimulationConfig.transmissionFile);
}

void EventAction::EndRun() {
  Output::Close(fExitOut, gSimulationConfig.exitKinematicsFile);
  Output::Close(fTransmissionOut, gSimulationConfig.transmissionFile);
}

void EventAction::BeginOfEventAction(const G4Event *) {
  fHasTritonExit = fGotMg = fGotS3Hit = false;
  fS3EnabledRingsHit = 0;
  fS3RingNumber = -1;
  // Missing kinematics have the same representation in ROOT and CSV.
  const auto nan = std::numeric_limits<G4double>::quiet_NaN();
  fRecoilEnergy = fRecoilTheta = fRecoilPhi = nan;
  fLightEnergy = fLightTheta = fLightPhi = nan;
  fS3SmearedKineticEnergy = fTritonEnergyAtCenter = fBeamEnergyAtCenter = nan;
}

void EventAction::StoreS3Multiplicity(G4int enabledRingCount) {
  fS3EnabledRingsHit = enabledRingCount;
}

void EventAction::StoreS3RingHit(G4int ringNumber, G4double smearedKineticEnergy,
                               G4double tritonEnergyAtCenter,
                               G4double beamEnergyAtCenter) {
  if (fS3EnabledRingsHit != 1 || fGotS3Hit || ringNumber < 1) {
    throw std::logic_error("S3 event summary requires exactly one enabled ring");
  }
  fGotS3Hit = true;
  fS3RingNumber = ringNumber;
  fS3SmearedKineticEnergy = smearedKineticEnergy;
  fTritonEnergyAtCenter = tritonEnergyAtCenter;
  fBeamEnergyAtCenter = beamEnergyAtCenter;
}

void EventAction::EndOfEventAction(const G4Event *event) {
  if (gSimulationMode != SimulationMode::Reaction) return;
  const bool passesS3Selection = fS3EnabledRingsHit == 1;
  if (passesS3Selection != fGotS3Hit) {
    throw std::logic_error("S3 multiplicity and selected hit disagree");
  }
  if (fRunAction) fRunAction->RecordS3Multiplicity(fS3EnabledRingsHit);
  auto *man = G4AnalysisManager::Instance();
  const auto &r = Analysis::Get().reaction;
  man->FillNtupleIColumn(r.id, r.eventID, event->GetEventID());
  man->FillNtupleDColumn(r.id, r.recoil.energy, fRecoilEnergy / MeV);
  man->FillNtupleDColumn(r.id, r.recoil.theta, fRecoilTheta / deg);
  man->FillNtupleDColumn(r.id, r.recoil.phi, fRecoilPhi / deg);
  man->FillNtupleDColumn(r.id, r.triton.energy, fLightEnergy / MeV);
  man->FillNtupleDColumn(r.id, r.triton.theta, fLightTheta / deg);
  man->FillNtupleDColumn(r.id, r.triton.phi, fLightPhi / deg);
  man->FillNtupleIColumn(r.id, r.hasRecoilExit, fGotMg);
  man->FillNtupleIColumn(r.id, r.hasTritonExit, fHasTritonExit);
  man->FillNtupleIColumn(r.id, r.enabledRingCount, fS3EnabledRingsHit);
  man->FillNtupleIColumn(r.id, r.passesS3Selection, passesS3Selection);
  man->AddNtupleRow(r.id);

  fExitOut << event->GetEventID() << ',' << gSimulationConfig.excitationEnergy / MeV
           << ',' << fGotMg << ',' << fRecoilEnergy / MeV << ',' << fRecoilTheta / deg
           << ',' << fRecoilPhi / deg << ',' << fHasTritonExit << ',' << fLightEnergy / MeV
           << ',' << fLightTheta / deg << ',' << fLightPhi / deg << ',';
  if (fGotS3Hit) {
    fExitOut << fS3RingNumber << ',' << fS3SmearedKineticEnergy / MeV << ','
             << fTritonEnergyAtCenter / MeV << ',' << fBeamEnergyAtCenter / MeV;
  } else {
    fExitOut << "nan,nan,nan,nan";
  }
  fExitOut << ',' << fS3EnabledRingsHit << ',' << passesS3Selection << '\n';
  Output::Check(fExitOut, gSimulationConfig.exitKinematicsFile);

  if (fGotMg && fHasTritonExit && passesS3Selection) {
    // These are projected recoil angles, not spherical theta and phi.
    const auto thetaSpec = std::atan(std::tan(fRecoilTheta) * std::cos(fRecoilPhi));
    const auto phiSpec = std::atan(std::tan(fRecoilTheta) * std::sin(fRecoilPhi));
    fTransmissionOut << fRecoilEnergy / MeV << ',' << thetaSpec / deg << ','
                     << phiSpec / deg << '\n';
    Output::Check(fTransmissionOut, gSimulationConfig.transmissionFile);
  }
}

void EventAction::StoreRecoil(G4double energy, G4double theta, G4double phi) {
  fGotMg = true;
  fRecoilEnergy = energy;
  fRecoilTheta = theta;
  fRecoilPhi = phi;
}

void EventAction::StoreLightParticle(G4double energy, G4double theta, G4double phi) {
  fHasTritonExit = true;
  fLightEnergy = energy;
  fLightTheta = theta;
  fLightPhi = phi;
}
