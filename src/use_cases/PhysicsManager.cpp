#include "PhysicsManager.h"
#include <chrono>

PhysicsManager::PhysicsManager(std::unique_ptr<IPhysicsEngine> engine)
    : engine(std::move(engine)), running(false)
{}

PhysicsManager::~PhysicsManager() {
    stop();
}

void PhysicsManager::start() {
    if (!running.load()) {
        running.store(true);
        engine->initialize();
        physicsThread = std::thread(&PhysicsManager::run, this);
    }
}

void PhysicsManager::stop() {
    if (running.load()) {
        running.store(false);
        if (physicsThread.joinable())
            physicsThread.join();
        engine->shutdown();
    }
}

void PhysicsManager::run() {
    using namespace std::chrono;
    auto previousTime = high_resolution_clock::now();
    while (running.load()) {
        auto currentTime = high_resolution_clock::now();
        float dt = duration<float>(currentTime - previousTime).count();
        previousTime = currentTime;
        engine->update(dt);
        // a little sleep to avoid cpu hogging
        std::this_thread::sleep_for(milliseconds(1));
    }
}
