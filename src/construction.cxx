#include "construction.hh"
#include "G4Colour.hh"
#include "G4NistManager.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "s3.hh"
#include <string>

// constructor
DetectorConstruction::DetectorConstruction()
    : fSiDeadLayerThickness(730 * nm), fLiFThickness(1.9 * um),
      fCThickness(0.245 * um), fTargetRadius(5.0 * mm),
      fTargetZOffset(0.25 * mm), fS3RingCount(24),
      fS3InnerRadius(11.0 * mm), fS3OuterRadius(35.0 * mm),
      fS3Thickness(308.0 * um), fS3DistanceFromTargetCenter(31.0 * mm) {}

void PlaceLayer(G4String name, G4Material *mat, G4double thickness,
                G4double targetRadius, G4double zCenter,
                G4LogicalVolume *logicWorld, G4VisAttributes *vis) {
  G4Tubs *solid = new G4Tubs(name + "_solid", 0.0, targetRadius, thickness / 2,
                             0.0, 360.0 * deg);
  G4LogicalVolume *logic = new G4LogicalVolume(solid, mat, name + "_logic");
  logic->SetVisAttributes(vis);
  new G4PVPlacement(nullptr, G4ThreeVector(0., 0., zCenter), logic,
                    name + "_phys", logicWorld, false, 0, true);
}

// implementing the method  in Detectorconstruction, it should return a
// PhysicalVolume pointer
G4VPhysicalVolume *DetectorConstruction::Construct() {
  G4NistManager *nist = G4NistManager::Instance();
  // use near vaccum as world volume:
  G4Material *worldMat = nist->FindOrBuildMaterial("G4_Galactic");
  // target: change for your experiment
  G4Material *Si = nist->FindOrBuildMaterial("G4_Si");
  G4Material *LiF = nist->FindOrBuildMaterial("G4_LITHIUM_FLUORIDE");
  G4Material *C = nist->FindOrBuildMaterial("G4_C");

  //  =============================== TARGET ==============================
  // Change geometry dimensions in the constructor initializer list.
  const G4double siThickness = fSiDeadLayerThickness;
  const G4double lifThickness = fLiFThickness;
  const G4double cThickness = fCThickness;
  const G4double targetRadius = fTargetRadius;
  const G4double zoffset = fTargetZOffset;
  
  G4double lifCenter = zoffset + lifThickness / 2;
  G4double cCenter = zoffset + lifThickness + cThickness / 2;

  // world volume:
  G4double worldSize = 10.0 * cm;
  G4Box *solidWorld = new G4Box("solidWorld", worldSize, worldSize, worldSize);
  G4LogicalVolume *logicWorld =
      new G4LogicalVolume(solidWorld, worldMat, "logicWorld");
  G4VPhysicalVolume *physWorld =
      new G4PVPlacement(nullptr, G4ThreeVector(0., 0., 0.), logicWorld,
                        "physWorld", nullptr, false, 0, true);

  auto *siVis = new G4VisAttributes(G4Colour(0.2, 0.6, 1.0, 0.45));
  siVis->SetForceSolid(true);

  auto *lifVis = new G4VisAttributes(G4Colour(0.2, 1.0, 0.4, 0.45));
  lifVis->SetForceSolid(true);

  auto* cVis = new G4VisAttributes(G4Colour(1.0, 0.2, 0.7, 0.6));
  cVis->SetForceSolid(true);

  PlaceLayer("LiF", LiF, lifThickness, targetRadius, lifCenter, logicWorld, lifVis);
  PlaceLayer("C", C, cThickness, targetRadius, cCenter, logicWorld, cVis);

  // ========================== S3 =============================
  fS3RingLogicals.clear();
  const G4int S3RingCount = fS3RingCount;
  const G4double S3InnerRadius = fS3InnerRadius;
  const G4double S3OuterRadius = fS3OuterRadius;
  const G4double s3Thickness = fS3Thickness;
  const G4double targetThickness = lifThickness + cThickness;
  const G4double targetCenterZ = zoffset + 0.5 * targetThickness;
  const G4double s3Z = targetCenterZ - fS3DistanceFromTargetCenter;
  const G4double ringWidth = (S3OuterRadius - S3InnerRadius) / S3RingCount;
  G4double siCenter = s3Z + s3Thickness / 2.0 + siThickness / 2.0; 
  auto *s3Vis = new G4VisAttributes(G4Colour(0.1, 0.7, 1.0, 0.35));
  s3Vis->SetForceSolid(true);
  auto *deadLayerVis = new G4VisAttributes(G4Colour(0.8, 0.8, 0.8, 0.35));
  deadLayerVis->SetForceSolid(true);

  auto *solidS3DeadLayer =
    new G4Tubs("s3DeadLayer_solid",
               S3InnerRadius,
               S3OuterRadius,
               siThickness / 2.0,
               0.0 * deg,
               360.0 * deg);

  auto *logicS3DeadLayer =
    new G4LogicalVolume(solidS3DeadLayer, Si, "s3DeadLayer_logic");

logicS3DeadLayer->SetVisAttributes(deadLayerVis);

new G4PVPlacement(nullptr,
                  G4ThreeVector(0.0, 0.0, siCenter),
                  logicS3DeadLayer,
                  "s3DeadLayer_phys",
                  logicWorld,
                  false,
                  0,
                  true);

  for (G4int ringID = 0; ringID < S3RingCount; ringID++) {
    const G4double ringInnerRadius = S3InnerRadius + ringID * ringWidth;
    const G4double ringOuterRadius = ringInnerRadius + ringWidth;

    G4Tubs *solidRing =
        new G4Tubs("s3Ring_solid_" + std::to_string(ringID), ringInnerRadius,
                   ringOuterRadius, s3Thickness / 2.0, 0.0 * deg, 360 * deg);

    G4LogicalVolume *logicRing = new G4LogicalVolume(
        solidRing, Si, "s3Ring_logic_" + std::to_string(ringID));
    logicRing->SetVisAttributes(s3Vis);
    // access ring number via:
    // preStepPoint->GetTouchableHandle()->GetCopyNumber();

    new G4PVPlacement(nullptr, G4ThreeVector(0.0, 0.0, s3Z), logicRing,
                      "s3Ring_phys_" + std::to_string(ringID), logicWorld,
                      false, ringID, true);

    fS3RingLogicals.push_back(logicRing);
  }

  return physWorld;
}

void DetectorConstruction::ConstructSDandField() {
  // attach a sensitive detector to each ring:
  auto *sdManager = G4SDManager::GetSDMpointer();
  auto *s3 = new S3("S3", fS3RingCount);
  sdManager->AddNewDetector(s3);
  for (auto *ringLogic : fS3RingLogicals) {
    ringLogic->SetSensitiveDetector(s3);
  }
}
