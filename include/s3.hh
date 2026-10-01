#ifndef S3_HH
#define S3_HH 

#include "G4VSensitiveDetector.hh"
#include "globals.hh"
#include <vector>

class G4HCofThisEvent; 
class G4Step; 
class G4TouchableHistory; 

class S3 : public G4VSensitiveDetector {
    public:
        S3(const G4String &name, G4int ringCount); 
        ~S3() override = default;
        void Initialize(G4HCofThisEvent *) override; 
        G4bool ProcessHits(G4Step * step, G4TouchableHistory *) override; 
        void EndOfEvent(G4HCofThisEvent *) override; 

    private: 
        struct RingHit { 
            G4bool hit = false; 
            G4double edep = 0.0; 
            G4double ekin = 0.0; 
            G4double theta = 0.0;
            G4double phi = 0.0; 
        }; 

        G4int fRingCount = 0; 
        std::vector<RingHit> fRingHits; 


}; 

#endif