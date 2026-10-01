#include "physics.hh"
#include "G4GenericIon.hh"
#include "G4EmParameters.hh"

PhysicsList::PhysicsList(){
    

    RegisterPhysics(new G4EmStandardPhysics()); 
    
    RegisterPhysics(new G4IonPhysics());
    RegisterPhysics(new G4OpticalPhysics());
    RegisterPhysics(new G4DecayPhysics());
    G4EmParameters::Instance()->SetBuildCSDARange(true);

}

PhysicsList::~PhysicsList(){}

void PhysicsList::ConstructParticle()
{
    G4VModularPhysicsList::ConstructParticle();
    G4GenericIon::GenericIonDefinition();
}