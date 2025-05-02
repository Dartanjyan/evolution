#include "PhysicsManager.h"
#include <chrono>

PhysicsManager::PhysicsManager(std::unique_ptr<IPhysicsEngine> engine)
    : engine(std::move(engine)), running(false)
{}


void PhysicsManager::getRenderObjects(std::vector<BodyObject> &bodies, std::vector<ShapeObject> &shapes, std::vector<ConstraintObject> &constraints) const
{
    if(running.load()) {
        engine->getRenderObjects(bodies, shapes, constraints);
    }
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
    
    // Constants for fixed timestep and target frame rate
    constexpr milliseconds TARGET_FRAME_TIME(16); // ~60 FPS (1000ms/60 ≈ 16.66ms)
    auto previous_time = high_resolution_clock::now();
    auto accumulated_lag = 0ms; // Tracks accumulated processing delays

    std::chrono::_V2::system_clock::time_point current_time;
    std::chrono::nanoseconds elapsed_time;

    while (running.load()) {
        // Measure elapsed time since last frame
        current_time = high_resolution_clock::now();
        elapsed_time = current_time - previous_time;
        previous_time = current_time;
        
        const float dt = 0.01f;
        engine->update(dt);

        // Handle all pending creature additions in this frame
        while (!creaturesQueue.empty()) {
            engine->addCreature(creaturesQueue.front());
            creaturesQueue.pop();
        }

        // Update rate regulation
        const auto processing_time = duration_cast<milliseconds>(
            high_resolution_clock::now() - current_time
        );

        // Calculate required sleep time to maintain target frame rate
        const auto sleep_time = TARGET_FRAME_TIME - processing_time;
        std::cout << sleep_time.count() << " ms" << std::endl;
        if (sleep_time > 0ms) {
            // If we have time left, sleep to maintain consistent frame rate
            std::this_thread::sleep_for(sleep_time);
            // accumulated_lag = 0ms; // Reset lag if we're on schedule
        } else {
            // If we're running behind, accumulate the delay
            // This will reduce sleep time in subsequent frames to catch up
            // accumulated_lag = -sleep_time;
        }
    }
}

