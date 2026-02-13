#ifndef GENERATION_MANAGER_H
#define GENERATION_MANAGER_H

#include <memory>

// #include "PhysicsManager.h"
class PhysicsManager;
// #include "DrawCommandCollector.h"
class DrawCommandCollector;

class GenerationManager {
public:
    GenerationManager(PhysicsManager* physicsManager, std::shared_ptr<DrawCommandCollector> drawCommandCollector, unsigned creaturesPerGeneration = 20, unsigned long ticksPerGeneration = 60*10);
    ~GenerationManager();
    
    void onTick(unsigned long tick);
    unsigned getGeneration() const { return generation; }
private:
    PhysicsManager* physicsManager;
    std::shared_ptr<DrawCommandCollector> drawCommandCollector;

    unsigned creaturesPerGeneration;
    unsigned long ticksPerGeneration;
    unsigned generation;

    void endGeneration();
};

#endif // GENERATION_MANAGER_H
