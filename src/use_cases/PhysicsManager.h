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
    ~PhysicsManager();

    // physics thread
    void start();
    void stop();

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) const;

    // Add new creature to the adding queue
    void addCreature(Creature* creature) { creaturesQueue.push(creature); }
    // Get necessary data from physics engine and send it to every Creature's Brain
    void updateCreaturesInputs();
};

#endif
