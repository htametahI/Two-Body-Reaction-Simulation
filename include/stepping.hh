#ifndef STEPPING_HH 
#define STEPPING_HH 

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include <fstream>


class EventAction; 

class SteppingAction : public G4UserSteppingAction {
    public:
        SteppingAction(EventAction* eventAction);

        ~SteppingAction() override = default; 
        void BeginRun();
        void EndRun();
        void UserSteppingAction(const G4Step* step) override; 

    private: 
        std::ofstream fBeamOut;
        EventAction *fEventAction = nullptr; 
        G4int fCurrentEventID = -1; 
        G4bool fWroteThisEvent = false; 

        G4double fReactionDepth = 0.0; 
        G4double fReactionZ = 0.0; 
};

#endif