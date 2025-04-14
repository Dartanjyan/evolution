#ifndef PHYSICS_MANAGER_H
#define PHYSICS_MANAGER_H

#include <memory>
#include <thread>
#include <atomic>
#include "IPhysicsEngine.h"

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

    const IPhysicsEngine* getEnginePtr() { return engine; }

private:
    void run();

    IPhysicsEngine* engine;
    std::thread physicsThread;
    std::atomic<bool> running;
};

#endif
