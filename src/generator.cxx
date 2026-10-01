#include "generator.hh"
#include "analysis.hh"
#include "output.hh"
#include "G4ParticleGun.hh"
#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4IonTable.hh"
#include "G4LorentzVector.hh"
#include "G4NucleiProperties.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"
#include "mode.hh"
#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>

// Input validation and stream setup happen before the first event of a run.
void PrimaryGenerator::BeginRun(G4int requestedEvents) {
  fNextBeamState = 0;
  fBeamStates.clear();
  fDwba = {};
  if (gSimulationMode != SimulationMode::Reaction) return;
  fBeamStates = ReadBeamFile(gSimulationConfig.beamFile);
  fDwba = ReadDwbaFile(gSimulationConfig.dwbaFile);
  if (requestedEvents >= 0 && static_cast<std::size_t>(requestedEvents) > fBeamStates.size()) {
    throw std::runtime_error("Requested more reaction events than rows in " +
                             std::string(gSimulationConfig.beamFile));
  }
  Output::Open(fTruthOut, gSimulationConfig.truthKinematicsFile);
  fTruthOut << "eventID,Ex_MeV,theta_cm_deg,phi_cm_deg,"
            << "Mg26_E_MeV,Mg26_theta_deg,Mg26_phi_deg,"
            << "triton_E_MeV,triton_theta_deg,triton_phi_deg,"
            << "beamRowID,beam_E_MeV,beam_x_mm,beam_y_mm,beam_z_mm,"
            << "beam_dirx,beam_diry,beam_dirz,depth_um\n";
  Output::Check(fTruthOut, gSimulationConfig.truthKinematicsFile);
}

void PrimaryGenerator::EndRun() {
  Output::Close(fTruthOut, gSimulationConfig.truthKinematicsFile);
}

// Sample DWBA: inverse transform method
G4double PrimaryGenerator::SampleDWBA() const {
  const G4double u = G4UniformRand();

  auto it = std::lower_bound(fDwba.cdf.begin(), fDwba.cdf.end(), u);
  if (it == fDwba.cdf.begin()) {
    return fDwba.theta.front();
  }

  const std::size_t i = std::distance(fDwba.cdf.begin(), it);

  const G4double c0 = fDwba.cdf[i - 1];
  const G4double c1 = fDwba.cdf[i];

  if (c1 <= c0) {
    return fDwba.theta[i];
  }

  const G4double f = (u - c0) / (c1 - c0);
  return fDwba.theta[i - 1] + f * (fDwba.theta[i] - fDwba.theta[i - 1]);
}

PrimaryGenerator::PrimaryGenerator() : fParticleGun(new G4ParticleGun(1)) {}
PrimaryGenerator::~PrimaryGenerator() { delete fParticleGun; }

// check for prepare or reaction mode, execute accordingly
void PrimaryGenerator::GeneratePrimaries(G4Event *anEvent) {
  if (gSimulationMode == SimulationMode::Prepare) {
    GenerateBeam(anEvent);
  } else {
    GenerateReaction(anEvent);
  }
}

void PrimaryGenerator::GenerateBeam(G4Event *event) {
  auto ion =
      G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(10, 22, 0.0);
  G4double sigmaE = gSimulationConfig.beam.fractionalEnergySpread * gSimulationConfig.beam.energy;
  // sample beam energy
  G4double energy = CLHEP::RandGauss::shoot(gSimulationConfig.beam.energy, sigmaE);
  // sample beam position from emittance
  G4double radius = gSimulationConfig.beam.spotDiameter / 2.0;
  G4double r = radius * std::sqrt(G4UniformRand());
  G4double spotPhi = CLHEP::twopi * G4UniformRand();
  G4double x = r * std::cos(spotPhi);
  G4double y = r * std::sin(spotPhi);

  // beam speed, needed for emittance
  G4double mass = ion->GetPDGMass();
  G4double gamma = (mass + energy) / mass;
  G4double beta = std::sqrt(1.0 - 1.0 / (gamma * gamma));

  G4double thetaMax = 0.0;
  if (radius > 0.0 && beta > 0.0) {
    thetaMax = gSimulationConfig.beam.normalizedEmittance / (beta * gamma * radius);
  }

  // theta spread around z axis, sqrt to sample a disk of angular directions
  //          theta_y
  //           |
  //      . . . . .
  //    .           .
  //   .      +      .  -> theta_x
  //    .           .
  //      . . . . .
  // u = theta^2 / thetaMax^2
  // theta = thetaMax * sqrt(u)

  G4double theta = thetaMax * std::sqrt(G4UniformRand());
  // phi spread
  G4double dirPhi = CLHEP::twopi * G4UniformRand();

  // beam direction
  G4ThreeVector direction(std::sin(theta) * std::cos(dirPhi),
                          std::sin(theta) * std::sin(dirPhi), std::cos(theta));

  // beam start point
  const G4double zStart = gSimulationConfig.beam.startZ;
  fParticleGun->SetParticleDefinition(ion);
  fParticleGun->SetParticleCharge(4.0 * eplus);
  fParticleGun->SetParticleEnergy(energy);
  fParticleGun->SetParticlePosition(G4ThreeVector(x, y, zStart));
  fParticleGun->SetParticleMomentumDirection(direction);
  fParticleGun->GeneratePrimaryVertex(event);
}

void PrimaryGenerator::GenerateReaction(G4Event *event) {
  if (fNextBeamState >= fBeamStates.size()) {
    G4Exception("PrimaryGenerator::GenerateReaction", "BeamFile003",
                FatalException, "Requested more events than rows in beam.dat.");
  }

  // read the next row of the beam file
  const BeamState &beam = fBeamStates[fNextBeamState++];

  // particles
  const G4int Z1 = 10, A1 = 22; // 22Ne
  const G4int Z2 = 3, A2 = 7;   // 7Li
  const G4int Z3 = 12, A3 = 26; // 26Mg
  const G4int Z4 = 1, A4 = 3;   // triton

  G4double Ebeam = beam.energy * MeV;
  G4ThreeVector beamDir(beam.ux, beam.uy, beam.uz);
  if (beamDir.mag2() > 0.0) {
    beamDir = beamDir.unit();
  }

  G4double m1 = G4NucleiProperties::GetNuclearMass(A1, Z1);
  G4double m2 = G4NucleiProperties::GetNuclearMass(A2, Z2);
  G4double m3 = G4NucleiProperties::GetNuclearMass(A3, Z3) +
                gSimulationConfig.excitationEnergy;
  G4double m4 = G4NucleiProperties::GetNuclearMass(A4, Z4);

  // beam momentum
  G4double p1mag = std::sqrt((m1 + Ebeam) * (m1 + Ebeam) - m1 * m1);
  // 4-vector of beam
  G4LorentzVector lv1(beamDir * p1mag, m1 + Ebeam);
  // 4-vector of 7Li
  G4LorentzVector lv2(0.0, 0.0, 0.0, m2);
  // sum
  G4LorentzVector total = lv1 + lv2;
  // velocity of CM relative to lab
  G4ThreeVector boost = total.boostVector();
  // boost into CM
  lv1.boost(-boost);
  lv2.boost(-boost);
  // total energy in CM
  G4double ecm = lv1.e() + lv2.e();

  // check if there is enough energy to create the products t + 26Mg
  if (ecm < m3 + m4) {
    G4Exception("PrimaryGenerator::GenerateReaction", "Reaction001",
                EventMustBeAborted, "Not enough energy for reaction.");
    return;
  }

  // energy of recoil (3) in CM
  G4double e3 = (ecm * ecm + m3 * m3 - m4 * m4) / (2.0 * ecm);
  // energy of ejectile (4) in CM
  G4double e4 = ecm - e3;
  // magnitude of momentum in the final state, both products are equal and
  // opposite
  G4double pcm = std::sqrt(e3 * e3 - m3 * m3);

  // sample DWBA results
  G4double theta = SampleDWBA();
  G4double cosTheta = std::cos(theta);
  G4double sinTheta = std::sin(theta);
  G4double phi = CLHEP::twopi * G4UniformRand();

  // The sampled triton CM polar axis points opposite to the incoming beam.
  G4ThreeVector uz = -beamDir.unit();
  G4ThreeVector ux = uz.orthogonal().unit();
  G4ThreeVector uy = uz.cross(ux).unit();

  // momentum vector of triton in CM
  G4ThreeVector p4cm = pcm * (sinTheta * std::cos(phi) * ux +
                              sinTheta * std::sin(phi) * uy + cosTheta * uz);

  // The 26Mg recoil balances the triton momentum in the CM frame.
  G4ThreeVector p3cm = -p4cm;

  // 4-vectors in CM
  G4LorentzVector lv3(p3cm, e3);
  G4LorentzVector lv4(p4cm, e4);

  // boost back to lab frame
  lv3.boost(boost);
  lv4.boost(boost);

  // reaction points from beam.dat
  G4ThreeVector vertex(beam.x * mm, beam.y * mm, beam.z * mm);

  auto mg26 = G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(
      Z3, A3, gSimulationConfig.excitationEnergy);

  auto triton =
      G4ParticleTable::GetParticleTable()->GetIonTable()->GetIon(Z4, A4, 0.0);

  G4ThreeVector p3(lv3.px(), lv3.py(), lv3.pz());
  G4ThreeVector p4(lv4.px(), lv4.py(), lv4.pz());

  // write out:
  const G4double mgEnergy = lv3.e() - m3;
  const G4double tEnergy = lv4.e() - m4;

  // Lab polar angles measured from the global +z axis.

  const G4double mgTheta = p3.theta();

  const G4double tTheta = p4.theta();

  G4double mgPhi = p3.phi();
  G4double tPhi = p4.phi();
  if (mgPhi < 0.0)
    mgPhi += 360.0 * deg;
  if (tPhi < 0.0)
    tPhi += 360.0 * deg;

  // write out info@reaction vertices
  fTruthOut << event->GetEventID() << ","
            << gSimulationConfig.excitationEnergy / MeV << ","
            << theta / deg << ","
            << phi / deg << ","
            << mgEnergy / MeV << "," << p3.theta() / deg << "," << mgPhi / deg
            << "," << tEnergy / MeV << "," << p4.theta() / deg << ","
            << tPhi / deg << "," << (fNextBeamState - 1) << ","
            << beam.energy << "," << beam.x << "," << beam.y << "," << beam.z << ","
            << beam.ux << "," << beam.uy << "," << beam.uz << "," << beam.depth << "\n";
  Output::Check(fTruthOut, gSimulationConfig.truthKinematicsFile);

  auto *analysis = G4AnalysisManager::Instance();

  analysis->FillH2(Analysis::Get().recoilTruth, mgTheta / deg, mgEnergy / MeV);
  analysis->FillH2(Analysis::Get().tritonTruth, tTheta / deg, tEnergy / MeV);

  fParticleGun->SetParticleDefinition(mg26);
  fParticleGun->SetParticleCharge(12.0 * eplus);
  fParticleGun->SetParticleEnergy((lv3.e() - m3));
  fParticleGun->SetParticlePosition(vertex);
  fParticleGun->SetParticleMomentumDirection(p3.unit());
  fParticleGun->GeneratePrimaryVertex(event);

  fParticleGun->SetParticleDefinition(triton);
  fParticleGun->SetParticleCharge(1.0 * eplus);
  fParticleGun->SetParticleEnergy((lv4.e() - m4));
  fParticleGun->SetParticlePosition(vertex);
  fParticleGun->SetParticleMomentumDirection(p4.unit());
  fParticleGun->GeneratePrimaryVertex(event);
}
