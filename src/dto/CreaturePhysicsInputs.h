#ifndef CREATUREPHYSICSINPUTS_H
#define CREATUREPHYSICSINPUTS_H

typedef struct CreaturePhysicsInputs {
public:
    Creature* creature;
    std::vector<float> features;
} CreaturePhysicsInputs;

#endif
