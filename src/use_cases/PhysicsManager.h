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

#define USE_HOOKS 0
#if USE_HOOKS
struct tickHook {
    unsigned long targetTick;
    std::condition_variable* cv;
};

// Comparator for std::multiset
struct TickHookCompare {
    bool operator()(const tickHook& a, const tickHook& b) const {
        return a.targetTick < b.targetTick;
    }
};
#endif

class PhysicsManager {
private:
    void run();

    #if USE_HOOKS
    void checkHooks(unsigned long currentTick);
    #endif

    std::queue<Creature*> creaturesQueue;

    std::unique_ptr<IPhysicsEngine> engine;
    std::unique_ptr<AIManager> ai_manager;
    std::thread physicsThread;
    std::atomic<bool> running;

    std::unique_ptr<GenerationManager> generationManager;

    #if USE_HOOKS
    std::multiset<tickHook, TickHookCompare> hooks;
    std::mutex hooksMutex;
    #endif

    unsigned long tickCounter;

    std::chrono::milliseconds updateInterval;
public:
    PhysicsManager(std::unique_ptr<IPhysicsEngine> engine, std::unique_ptr<IAICalculator> ai_calculator);
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
    #if USE_HOOKS
    /*
     * struct tickHook {unsigned long targetTick; std::condition_variable* cv; };
     * Add new hook that'll fire conditional_variable after targetTick ticks.
     */
    void addHook(tickHook& newHook);
    #endif

    void setUpdateTimeScale(float scale);
};

#endif
