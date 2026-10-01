#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4SystemOfUnits.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4PVPlacement.hh"
#include <vector>


class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override = default;
    G4VPhysicalVolume* Construct() override;

    G4double GetSiDeadLayerThickness() const { return fSiDeadLayerThickness; }
    G4double GetLiFThickness() const { return fLiFThickness; }
    G4double GetCThickness() const { return fCThickness; }
    G4double GetTargetRadius() const { return fTargetRadius; }
    G4double GetTargetZOffset() const { return fTargetZOffset; }
    G4double GetLiFFrontZ() const { return fTargetZOffset; }

    G4int GetS3RingCount() const { return fS3RingCount; }
    G4double GetS3InnerRadius() const { return fS3InnerRadius; }
    G4double GetS3OuterRadius() const { return fS3OuterRadius; }
    G4double GetS3Thickness() const { return fS3Thickness; }
    G4double GetS3DistanceFromTargetCenter() const {
      return fS3DistanceFromTargetCenter;
    }

  private: 
    std::vector<G4LogicalVolume*> fS3RingLogicals;  

    G4double fSiDeadLayerThickness = 0.0;
    G4double fLiFThickness = 0.0;
    G4double fCThickness = 0.0;
    G4double fTargetRadius = 0.0;
    G4double fTargetZOffset = 0.0;

    G4int fS3RingCount = 0;
    G4double fS3InnerRadius = 0.0;
    G4double fS3OuterRadius = 0.0;
    G4double fS3Thickness = 0.0;
    G4double fS3DistanceFromTargetCenter = 0.0;

    virtual void ConstructSDandField() override; 
    
    
};

#endif
