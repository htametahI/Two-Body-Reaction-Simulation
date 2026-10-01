#include "action.hh"
#include "generator.hh"
#include "event.hh"
#include "stepping.hh"
#include "run.hh"

ActionInitialization::ActionInitialization() = default;
ActionInitialization::~ActionInitialization() = default;

void ActionInitialization::Build() const {
  auto *run = new RunAction();
  auto *generator = new PrimaryGenerator();
  auto *event = new EventAction(run);
  auto *stepping = new SteppingAction(event);
  run->SetActions(generator, event, stepping);
  SetUserAction(generator);
  SetUserAction(run);
  SetUserAction(event);
  SetUserAction(stepping);
}
