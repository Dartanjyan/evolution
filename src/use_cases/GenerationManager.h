#ifndef GENERATION_MANAGER_H
#define GENERATION_MANAGER_H

// #include "PhysicsManager.h"
class PhysicsManager;

class GenerationManager {
public:
    GenerationManager(PhysicsManager* physicsManager, unsigned creaturesPerGeneration = 20, unsigned long ticksPerGeneration = 60*10);
    ~GenerationManager();
    
    void onTick(unsigned long tick);
    unsigned getGeneration() const { return generation; }
private:
    PhysicsManager* physicsManager;

    unsigned creaturesPerGeneration;
    unsigned long ticksPerGeneration;
    unsigned generation;

    void endGeneration();
};

#endif // GENERATION_MANAGER_H