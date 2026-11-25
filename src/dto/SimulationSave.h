#ifndef SIMULATION_SAVE
#define SIMULATION_SAVE

#include "Creature.h"
#include <memory>
#include <vector>

struct SimulationSave {
    // Maybe simulation settings like mutation algorithm, amount of creatures etc
    unsigned generation;    
    std::vector<Creature*> creatures;
};

struct LoadedSimulation {
    unsigned generation;
    std::vector<std::unique_ptr<Creature>> creatures;
};

#endif
