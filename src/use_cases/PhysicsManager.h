#ifndef PHYSICS_MANAGER_H
#define PHYSICS_MANAGER_H

#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <set>
#include <chrono>
#include <condition_variable>
#include "IPhysicsEngine.h"
#include "IAICalculator.h"
#include "AIManager.h"
#include "Creature.h"
#include "GenerationManager.h"
#include "ISimulationSaver.h"

class PhysicsManager {
private:
    void run();

    std::queue<Creature*> creaturesQueue;

    std::unique_ptr<GenerationManager> generationManager;
    std::unique_ptr<IPhysicsEngine> engine;
    std::unique_ptr<AIManager> ai_manager;
    std::thread physicsThread;
    std::atomic<bool> running;
    std::unique_ptr<ISimulationSaver> simulationSaver;

    unsigned long tickCounter;

    std::chrono::milliseconds updateInterval;
public:
    PhysicsManager(std::unique_ptr<IPhysicsEngine> engine, std::unique_ptr<IAICalculator> ai_calculator, std::unique_ptr<ISimulationSaver> simulationSaver);
    ~PhysicsManager();

    // physics thread
    void start();
    void stop();
    void stopInternal();
    void startInternal();

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) const;

    // Add new creature to the adding queue
    void addCreature(Creature* creature) { creaturesQueue.push(creature); }
    void removeCreature(Creature* creature);
    void removeAllCreatures();

    void getCreatures(std::vector<Creature *>& out);
    unsigned int getGeneration();

    void setUpdateTimeScale(float scale);
};

#endif
