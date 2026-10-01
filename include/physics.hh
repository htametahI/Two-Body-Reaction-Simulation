#ifndef PHYSICS_HH
#define PHYSICS_HH 

#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4OpticalPhysics.hh"
#include "G4DecayPhysics.hh"


#include "G4IonPhysics.hh"

class PhysicsList : public G4VModularPhysicsList
{
    public:
        PhysicsList();
        ~PhysicsList();
        virtual void ConstructParticle() override;

    
};

#endif
