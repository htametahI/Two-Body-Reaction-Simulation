#include "G4RunManager.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "Randomize.hh"
#include "action.hh"
#include "construction.hh"
#include "input.hh"
#include "mode.hh"
#include "output.hh"
#include "physics.hh"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Usage() {
  std::cout << "Usage:\n"
            << "  sim [--seed N] [--beam-file PATH]\n"
            << "  sim --prepare [nEvents] [--seed N] [--beam-file PATH]\n"
            << "  sim --reaction DWBA Ex_MeV OUTPUT.root [nEvents] [--seed N] [--beam-file PATH]\n"
            << "  sim --vis [--seed N] [--beam-file PATH]\n"
            << "  sim MACRO [--seed N] [--beam-file PATH]\n"
            << "  --rng-state PATH restores a saved full run-start state instead of --seed.\n";
}

long PositiveInteger(const std::string &text, const std::string &name) {
  std::size_t used = 0;
  const long value = std::stol(text, &used);
  if (used != text.size() || value <= 0) throw std::runtime_error(name + " must be a positive integer");
  return value;
}

G4int EventCount(const std::string &text) {
  const auto value = PositiveInteger(text, "nEvents");
  if (value > std::numeric_limits<G4int>::max()) throw std::runtime_error("nEvents exceeds G4int range");
  return static_cast<G4int>(value);
}

G4int BeamRows() {
  const auto count = ReadBeamFile(gSimulationConfig.beamFile).size();
  if (count > static_cast<std::size_t>(std::numeric_limits<G4int>::max())) {
    throw std::runtime_error("Beam file has too many rows for one run");
  }
  return static_cast<G4int>(count);
}

void RestoreRandomState(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open random-state file: " + path);
  CLHEP::HepRandom::restoreFullState(input);
  if (input.fail()) throw std::runtime_error("Invalid random-state file: " + path);
}
}

int main(int argc, char **argv) {
  try {
    std::vector<std::string> args;
    std::string randomState;
    bool seedSet = false;
    // Parse common options before creating geometry or opening any output files.
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") { Usage(); return 0; }
      if (arg == "--seed" || arg == "--beam-file" || arg == "--rng-state") {
        if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
        if (arg == "--seed") {
          CLHEP::HepRandom::setTheSeed(PositiveInteger(argv[i], "seed"));
          seedSet = true;
        } else if (arg == "--beam-file") gSimulationConfig.beamFile = argv[i];
        else randomState = argv[i];
      } else args.push_back(arg);
    }
    if (seedSet && !randomState.empty()) throw std::runtime_error("Choose either --seed or --rng-state");
    const std::string command = args.empty() ? "" : args.front();
    G4int events = 1000;
    if (command == "--reaction") {
      if (args.size() != 4 && args.size() != 5) throw std::runtime_error("--reaction requires DWBA Ex_MeV OUTPUT.root [nEvents]");
      gSimulationConfig.dwbaFile = args[1];
      std::size_t used = 0;
      const double excitation = std::stod(args[2], &used);
      if (used != args[2].size() || !std::isfinite(excitation) || excitation < 0.0) {
        throw std::runtime_error("Ex_MeV must be finite and nonnegative");
      }
      gSimulationConfig.excitationEnergy = excitation * MeV;
      gSimulationConfig.outputFile = args[3];
      gSimulationConfig.truthKinematicsFile = Output::WithSuffix(args[3], "_truth_kinematics.csv");
      gSimulationConfig.exitKinematicsFile = Output::WithSuffix(args[3], "_exit_kinematics.csv");
      auto state = std::filesystem::path(args[1]).parent_path().filename().string();
      if (state.empty()) state = "reaction";
      gSimulationConfig.transmissionFile = "../output/transmission/" + state + ".csv";
      events = args.size() == 5 ? EventCount(args[4]) : BeamRows();
    } else if (command == "--prepare") {
      if (args.size() > 2) throw std::runtime_error("--prepare accepts only [nEvents]");
      if (args.size() == 2) events = EventCount(args[1]);
    } else if (command == "--vis") {
      if (args.size() != 1) throw std::runtime_error("--vis accepts no positional arguments");
    } else if (!command.empty() && (command.front() == '-' || args.size() != 1)) {
      throw std::runtime_error("Unknown command or extra arguments: " + command);
    }
    ValidateSimulationConfig();

    auto manager = std::make_unique<G4RunManager>();
    manager->SetUserInitialization(new DetectorConstruction());
    manager->SetUserInitialization(new PhysicsList());
    manager->SetUserInitialization(new ActionInitialization());
    manager->Initialize();
    auto vis = std::make_unique<G4VisExecutive>();
    vis->Initialise();
    // Restore after initialization, immediately before the first run. This is
    // the same boundary at which RunAction saves the complete random state.
    if (!randomState.empty()) RestoreRandomState(randomState);
    auto *ui = G4UImanager::GetUIpointer();
    ui->ApplyCommand("/run/printProgress 100");
    const auto run = [&](SimulationMode mode, G4int count) {
      gSimulationMode = mode;
      manager->BeamOn(count);
    };

    if (command == "--prepare") run(SimulationMode::Prepare, events);
    else if (command == "--reaction") run(SimulationMode::Reaction, events);
    else if (command.empty() || command == "--vis") {
      run(SimulationMode::Prepare, events);
      const auto reactionEvents = BeamRows();
      if (command == "--vis") {
        G4UIExecutive session(argc, argv);
        gSimulationMode = SimulationMode::Reaction;
        if (ui->ApplyCommand("/control/execute vis.mac") != 0) throw std::runtime_error("Failed executing vis.mac");
        run(SimulationMode::Reaction, std::min(10000, reactionEvents));
        session.SessionStart();
      } else run(SimulationMode::Reaction, reactionEvents);
    } else {
      gSimulationMode = SimulationMode::Prepare;
      if (ui->ApplyCommand("/control/execute " + command) != 0) {
        throw std::runtime_error("Failed executing macro: " + command);
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "Simulation error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
