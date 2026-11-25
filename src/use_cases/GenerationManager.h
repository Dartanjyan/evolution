#ifndef GENERATION_MANAGER_H
#define GENERATION_MANAGER_H

#include <atomic>

// #include "PhysicsManager.h"
class PhysicsManager;
// #include "DrawCommandCollector.h"
class DrawCommandCollector;

class GenerationManager {
public:
    GenerationManager(PhysicsManager* physicsManager, DrawCommandCollector* drawCommandCollector, unsigned creaturesPerGeneration = 20, unsigned long ticksPerGeneration = 60*10);
    ~GenerationManager();
    
    void onTick(unsigned long tick);
    unsigned getGeneration() const { return generation.load(); }
    void restoreGeneration(unsigned number, unsigned populationSize);
    
    void updateScreenInfo();
    bool screenInfoInitialized = false;
private:
    PhysicsManager* physicsManager;
    DrawCommandCollector* drawCommandCollector;

    unsigned creaturesPerGeneration;
    unsigned long ticksPerGeneration;
    std::atomic<unsigned> generation;

    void endGeneration();
};

#endif // GENERATION_MANAGER_H
