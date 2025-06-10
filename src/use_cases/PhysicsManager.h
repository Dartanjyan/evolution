#ifndef PHYSICS_MANAGER_H
#define PHYSICS_MANAGER_H

#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include "IPhysicsEngine.h"
#include "IAICalculator.h"
#include "AIManager.h"
#include "Creature.h"

class PhysicsManager {
private:
    void run();

    std::queue<Creature*> creaturesQueue;

    std::unique_ptr<IPhysicsEngine> engine;
    std::unique_ptr<AIManager> ai_manager;
    std::thread physicsThread;
    std::atomic<bool> running;
public:
    PhysicsManager(std::unique_ptr<IPhysicsEngine> engine, std::unique_ptr<IAICalculator> ai_calculator);
    PhysicsManager(std::unique_ptr<IPhysicsEngine> engine);
    ~PhysicsManager();

    // physics thread
    void start();
    void stop();

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) const;

    // A function to add creature to a queue of adding chipmunkCreatures
    void addCreature(Creature* creature) { creaturesQueue.push(creature); }
};

#endif
