#ifndef CREATUREPHYSICSINPUTS_H
#define CREATUREPHYSICSINPUTS_H

#include "Creature.h"

typedef struct CreaturePhysicsInputs {
public:
    Creature* creature;
    std::vector<double> inputs;
    std::vector<double> outputs;
} CreaturePhysicsInputs;

#endif
