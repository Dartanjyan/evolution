#ifndef PHYSICS_MANAGER_H
#define PHYSICS_MANAGER_H

#include <memory>
#include <thread>
#include <atomic>
#include "IPhysicsEngine.h"
#include "Creature.h"
#include <queue>

class PhysicsManager {
public:
    PhysicsManager(std::unique_ptr<IPhysicsEngine> engine);
    ~PhysicsManager();

    // physics thread
    void start();
    void stop();

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) const;

    // const std::unique_ptr<IPhysicsEngine> getEnginePtr() const { return engine; }

    // A function to add creature to a queue of adding chipmunkCreatures
    void addCreature(Creature* creature) { creaturesQueue.push(creature); }

private:
    void run();

    std::queue<Creature*> creaturesQueue;

    std::unique_ptr<IPhysicsEngine> engine;
    std::thread physicsThread;
    std::atomic<bool> running;
};

#endif
