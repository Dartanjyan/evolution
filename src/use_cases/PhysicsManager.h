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
    PhysicsManager(IPhysicsEngine* engine);
    ~PhysicsManager();

    // physics thread
    void start();
    void stop();

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) const;

    const IPhysicsEngine* getEnginePtr() const { return engine; }

    // A function to add creature to a queue of adding creatures
    void addCreature(Creature* creature) { creaturesQueue.push(creature); }

private:
    void run();

    std::queue<Creature*> creaturesQueue;

    IPhysicsEngine* engine;
    std::thread physicsThread;
    std::atomic<bool> running;
};

#endif
