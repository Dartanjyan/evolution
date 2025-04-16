#include "PhysicsManager.h"
#include <chrono>

PhysicsManager::PhysicsManager(IPhysicsEngine* engine)
    : engine(engine), running(false)
{}


void PhysicsManager::getRenderObjects(std::vector<BodyObject> &bodies, std::vector<ShapeObject> &shapes, std::vector<ConstraintObject> &constraints) const
{
    engine->getRenderObjects(bodies, shapes, constraints);
}

PhysicsManager::~PhysicsManager() {
    stop();
}

void PhysicsManager::start() {
    if (!running.load()) {
        running.store(true);
        engine->initialize();
        physicsThread = std::thread(&PhysicsManager::run, this);
        std::cout << "Created new physics thread\n";
    } else {
        std::cout << "Physics thread already running\n";
    }
}

void PhysicsManager::stop() {
    if (running.load()) {
        running.store(false);
        if (physicsThread.joinable())
            physicsThread.join();
        engine->shutdown();
        std::cout<<"Physics engine has been shut down\n";
    }
}

void PhysicsManager::run() {
    using namespace std::chrono;
    auto previousTime = high_resolution_clock::now();

    // NOTE: probably atomic bottleneck
    // p.s. oh this is not in engine's thread
    while (running.load()) {
        auto currentTime = high_resolution_clock::now();
        float dt = duration<float>(currentTime - previousTime).count();
        previousTime = currentTime;
        engine->update(dt);
	
	while (!this->creaturesQueue.empty()) {
	    engine->addCreature(this->creaturesQueue.front());
	    creaturesQueue.pop();
	}
        // a little sleep to avoid cpu hogging
        std::this_thread::sleep_for(milliseconds(1));
    }
}

