#ifndef PHYSICS_MANAGER_H
#define PHYSICS_MANAGER_H

#include <memory>
#include <thread>
#include <atomic>
#include "IPhysicsEngine.h"

class PhysicsManager {
public:
    PhysicsManager(std::unique_ptr<IPhysicsEngine> engine);
    ~PhysicsManager();

    // physics thread
    void start();
    void stop();

private:
    void run();

    std::unique_ptr<IPhysicsEngine> engine;
    std::thread physicsThread;
    std::atomic<bool> running;
};

#endif
