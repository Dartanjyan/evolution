#ifndef CREATUREPHYSICSINPUTS_H
#define CREATUREPHYSICSINPUTS_H

#include "Creature.h"

typedef struct CreaturePhysicsInputs {
public:
    Creature* creature;
    std::vector<float> inputs;
    std::vector<float> outputs;
} CreaturePhysicsInputs;

#endif
