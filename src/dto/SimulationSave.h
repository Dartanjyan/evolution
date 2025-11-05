#ifndef SIMULATION_SAVE
#define SIMULATION_SAVE

#include "Creature.h"
#include <vector>

struct SimulationSave {
    // Maybe simulation settings like mutation algorithm, amount of creatures etc
    unsigned generation;    
    std::vector<Creature*> creatures;
};

#endif
