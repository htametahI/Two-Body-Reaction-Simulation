#include "run.hh"
#include "analysis.hh"
#include "construction.hh"
#include "event.hh"
#include "generator.hh"
#include "mode.hh"
#include "output.hh"
#include "stepping.hh"
#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4Version.hh"
#include "Randomize.hh"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
void SaveRandomState(const std::string &path) {
  std::ofstream stream;
  Output::Open(stream, path);
  // The full state includes distribution caches; recording a seed alone is
  // insufficient to restart a reaction run following beam preparation.
  CLHEP::HepRandom::saveFullState(stream);
  Output::Close(stream, path);
}

std::string Absolute(const std::string &path) {
  return std::filesystem::absolute(path).lexically_normal().string();
}
}

RunAction::RunAction() { Analysis::Book(); }

void RunAction::SetActions(PrimaryGenerator *generator, EventAction *event,
                           SteppingAction *stepping) {
  fGenerator = generator;
  fEvent = event;
  fStepping = stepping;
}

void RunAction::RecordS3Multiplicity(int enabledRingCount) {
  if (enabledRingCount < 0) throw std::logic_error("Negative S3 multiplicity");
  ++fS3Counts[enabledRingCount == 0 ? 0 : enabledRingCount == 1 ? 1 : 2];
}

void RunAction::BeginOfRunAction(const G4Run *run) {
  ValidateSimulationConfig();
  fS3Counts.fill(0);
  const auto &c = gSimulationConfig;
  const bool reaction = gSimulationMode == SimulationMode::Reaction;
  const std::string base = reaction ? c.outputFile : c.beamFile;
  fMetadataPath = Output::WithSuffix(base, "_metadata.json");
  const auto randomStart = Output::WithSuffix(base, "_random_start.rndm");
  fRandomEndPath = Output::WithSuffix(base, "_random_end.rndm");

  std::ostringstream meta;
  meta << std::setprecision(17);
  meta << "{\n  \"schema_version\": 2,\n  \"mode\": "
       << Output::JsonString(reaction ? "reaction" : "prepare")
       << ",\n  \"run_id\": " << run->GetRunID()
       << ",\n  \"started_unix_seconds\": " << std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()
       << ",\n  \"geant4_version\": " << Output::JsonString(G4Version)
       << ",\n  \"build\": " << Output::JsonString(__DATE__ " " __TIME__)
       << ",\n  \"random\": {\"engine\": " << Output::JsonString(CLHEP::HepRandom::getTheEngine()->name())
       << ", \"seed\": " << CLHEP::HepRandom::getTheSeed()
       << ", \"start_state\": " << Output::JsonString(Absolute(randomStart))
       << ", \"end_state\": " << Output::JsonString(Absolute(fRandomEndPath)) << "},"
       << "\n  \"beam_settings\": {\"energy_MeV\": " << c.beam.energy / MeV
       << ", \"fractional_energy_sigma\": " << c.beam.fractionalEnergySpread
       << ", \"spot_diameter_mm\": " << c.beam.spotDiameter / mm
       << ", \"normalized_emittance_mm_mrad\": " << c.beam.normalizedEmittance / (mm * mrad)
       << ", \"start_z_mm\": " << c.beam.startZ / mm << "},"
       << "\n  \"Ex_MeV\": " << c.excitationEnergy / MeV
       << ",\n  \"reconstruction\": {\"s3_energy_sigma_keV\": " << c.reconstruction.s3EnergySigma / keV
       << ", \"assumed_beam_energy_at_center_MeV\": " << c.reconstruction.assumedBeamEnergyAtCenter / MeV
       << ", \"beam_energy_reference\": \"legacy fixed value; original calibration provenance unavailable\""
       << ", \"triton_depth_assumption\": \"half LiF thickness\", \"angle_assumption\": \"ring center\"},"
       << "\n  \"selection\": {\"required_enabled_rings\": 1, \"excluded_ring_numbers\": [";
  for (std::size_t i = 0; i < c.excludedS3RingNumbers.size(); ++i) {
    if (i) meta << ',';
    meta << c.excludedS3RingNumbers[i];
  }
  meta << "], \"ring_number_base\": 1, \"deposit_threshold_MeV\": 0,"
       << " \"primary_tritons_only\": true},";
  const auto *detector = dynamic_cast<const DetectorConstruction *>(
      G4RunManager::GetRunManager()->GetUserDetectorConstruction());
  if (!detector) throw std::logic_error("Run metadata requires DetectorConstruction");
  for (auto number : c.excludedS3RingNumbers) {
    if (number < 1 || number > detector->GetS3RingCount()) {
      throw std::runtime_error("Excluded S3 ring number is outside the detector");
    }
  }
  meta << "\n  \"geometry\": {\"LiF_thickness_um\": " << detector->GetLiFThickness() / um
       << ", \"carbon_thickness_um\": " << detector->GetCThickness() / um
       << ", \"silicon_dead_layer_nm\": " << detector->GetSiDeadLayerThickness() / nm
       << ", \"target_radius_mm\": " << detector->GetTargetRadius() / mm
       << ", \"LiF_front_z_mm\": " << detector->GetLiFFrontZ() / mm
       << ", \"S3_ring_count\": " << detector->GetS3RingCount()
       << ", \"S3_inner_radius_mm\": " << detector->GetS3InnerRadius() / mm
       << ", \"S3_outer_radius_mm\": " << detector->GetS3OuterRadius() / mm
       << ", \"S3_thickness_um\": " << detector->GetS3Thickness() / um
       << ", \"S3_distance_from_target_center_mm\": " << detector->GetS3DistanceFromTargetCenter() / mm << "},"
       << "\n  \"inputs\": ";
  if (reaction && fGenerator) {
    meta << "{\"beam\": " << Output::FileIdentityJson(c.beamFile)
         << ", \"dwba\": " << Output::FileIdentityJson(c.dwbaFile) << '}';
  } else {
    // Custom primary generators (e.g. regression fixtures) do not read these inputs.
    meta << "null";
  }
  meta << ",\n  \"outputs\": {\"beam\": " << Output::JsonString(Absolute(c.beamFile));
  if (reaction) {
    meta << ", \"root\": " << Output::JsonString(Absolute(c.outputFile))
         << ", \"truth_csv\": " << Output::JsonString(Absolute(c.truthKinematicsFile))
         << ", \"exit_csv\": " << Output::JsonString(Absolute(c.exitKinematicsFile))
         << ", \"transmission_csv\": " << Output::JsonString(Absolute(c.transmissionFile));
  }
  meta << "},\n";
  fMetadataPrefix = meta.str();

  // Load and validate all reaction input before processing events. Text outputs
  // are recreated per run, consistently with the ROOT output lifecycle.
  if (fGenerator) fGenerator->BeginRun(run->GetNumberOfEventToBeProcessed());
  if (fEvent) fEvent->BeginRun();
  if (fStepping) fStepping->BeginRun();
  if (reaction) {
    Output::EnsureParentDirectory(c.outputFile);
    if (!G4AnalysisManager::Instance()->OpenFile(c.outputFile)) {
      throw std::runtime_error("Cannot open ROOT output: " + std::string(c.outputFile));
    }
  }
  SaveRandomState(randomStart);
  WriteMetadata(run, "running");
}

void RunAction::WriteMetadata(const G4Run *run, const std::string &status) {
  std::ofstream output;
  Output::Open(output, fMetadataPath);
  output << fMetadataPrefix
         << "  \"status\": " << Output::JsonString(status)
         << ",\n  \"requested_events\": " << run->GetNumberOfEventToBeProcessed()
         << ",\n  \"processed_events\": " << run->GetNumberOfEvent()
         << ",\n  \"s3_event_counts\": {\"zero_enabled_rings\": " << fS3Counts[0]
         << ", \"one_enabled_ring\": " << fS3Counts[1]
         << ", \"multiple_enabled_rings\": " << fS3Counts[2] << "}";
  if (gSimulationMode == SimulationMode::Prepare && status == "complete") {
    output << ",\n  \"prepared_beam\": " << Output::FileIdentityJson(gSimulationConfig.beamFile);
  }
  output << "\n}\n";
  Output::Close(output, fMetadataPath);
}

void RunAction::EndOfRunAction(const G4Run *run) {
  if (fGenerator) fGenerator->EndRun();
  if (fEvent) fEvent->EndRun();
  if (fStepping) fStepping->EndRun();
  if (gSimulationMode == SimulationMode::Reaction) {
    auto *man = G4AnalysisManager::Instance();
    const bool written = man->Write();
    const bool closed = man->CloseFile();
    if (!written || !closed) throw std::runtime_error("Failed saving ROOT output: " + std::string(gSimulationConfig.outputFile));
    G4cout << "S3 enabled-ring multiplicity: zero=" << fS3Counts[0]
           << ", one=" << fS3Counts[1] << ", multiple=" << fS3Counts[2] << G4endl;
  }
  SaveRandomState(fRandomEndPath);
  WriteMetadata(run, "complete");
}
