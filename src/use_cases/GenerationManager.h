#ifndef GENERATION_MANAGER_H
#define GENERATION_MANAGER_H

// #include "PhysicsManager.h"
class PhysicsManager;

class GenerationManager {
public:
    GenerationManager(PhysicsManager* physicsManager, unsigned long ticksPerGeneration = 60*10, unsigned creaturesPerGeneration = 20);

    void onTick(unsigned long tick);
    unsigned getGeneration() const { return generation; }
private:
    PhysicsManager* physicsManager;

    unsigned long ticksPerGeneration;
    unsigned creaturesPerGeneration;
    unsigned generation;

    void endGeneration();
};

#endif // GENERATION_MANAGER_H