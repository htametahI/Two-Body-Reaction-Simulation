#include "construction.hh"
#include "generator.hh"
#include "mode.hh"
#include "output.hh"
#include "physics.hh"
#include "run.hh"
#include "G4Event.hh"
#include "G4NucleiProperties.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    gSimulationMode = SimulationMode::Reaction;
    gSimulationConfig.beamFile = "beam.dat";
    gSimulationConfig.dwbaFile = "dwba.dat";
    gSimulationConfig.truthKinematicsFile = "truth.csv";
    std::ofstream out;
    Output::Open(out, "beam.dat");
    for (int i = 0; i < 12; ++i) {
      out << 60. + i << " 0.1 0.2 0.251 " << (i - 6) * 0.003 << " 0.005 1 1\n";
    }
    Output::Close(out, "beam.dat");
    Output::Open(out, "dwba.dat");
    out << "0 1\n30 1\n90 1\n150 1\n180 1\n";
    Output::Close(out, "dwba.dat");
    G4RunManager manager;
    manager.SetUserInitialization(new DetectorConstruction());
    manager.SetUserInitialization(new PhysicsList());
    auto *generator = new PrimaryGenerator();
    manager.SetUserAction(generator);
    manager.SetUserAction(new RunAction());
    manager.Initialize();
    generator->BeginRun(12);
    CLHEP::HepRandom::setTheSeed(1729);
    const auto states = ReadBeamFile("beam.dat");
    const auto beamMass = G4NucleiProperties::GetNuclearMass(22, 10);
    const auto targetMass = G4NucleiProperties::GetNuclearMass(7, 3);
    for (int i = 0; i < 12; ++i) {
      G4Event event(i);
      generator->GeneratePrimaries(&event);
      if (event.GetNumberOfPrimaryVertex() != 2) throw std::runtime_error("Expected both products");
      const auto *recoil = event.GetPrimaryVertex(0)->GetPrimary();
      const auto *triton = event.GetPrimaryVertex(1)->GetPrimary();
      const auto &b = states[i];
      const auto energy = b.energy * MeV;
      const auto momentum = G4ThreeVector(b.ux, b.uy, b.uz).unit() *
          std::sqrt((beamMass + energy) * (beamMass + energy) - beamMass * beamMass);
      const auto deltaE = recoil->GetTotalEnergy() + triton->GetTotalEnergy() -
          (beamMass + energy + targetMass);
      const auto deltaP = recoil->GetMomentum() + triton->GetMomentum() - momentum;
      if (std::abs(deltaE) > 1e-6 * MeV || deltaP.mag() > 1e-6 * MeV) {
        throw std::runtime_error("Generated primaries violate four-momentum conservation");
      }
    }
    generator->EndRun();
    std::cout << "PASS: four-momentum conservation for 12 energies/directions\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
