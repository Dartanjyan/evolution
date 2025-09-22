#include "PhysicsManager.h"
#include <chrono>
#include "AIManager.h"
#include "BrainEditor.h"

// TODO: Call this somewhere on simulation start
// Creature::resetId();
// BodyPart::resetId();
// Constraint::resetId();
// Brain::resetId();

PhysicsManager::PhysicsManager(std::unique_ptr<IPhysicsEngine> engine, std::unique_ptr<IAICalculator> ai_calculator)
    : engine(std::move(engine)), running(false), tickCounter(0)
{
    ai_manager = std::make_unique<AIManager>(std::move(ai_calculator));
}

void PhysicsManager::getRenderObjects(std::vector<BodyObject> &bodies, std::vector<ShapeObject> &shapes, std::vector<ConstraintObject> &constraints) const
{
    if(running.load()) {
        engine->getRenderObjects(bodies, shapes, constraints);
    }
}

void PhysicsManager::getCreatures(std::vector<Creature *>& out)
{
    engine->getCreatures(out);
}

PhysicsManager::~PhysicsManager() {
    stop();
}

void PhysicsManager::start() {
    generationManager = std::make_unique<GenerationManager>(this);
    if (!running.load()) {
        running.store(true);
        if (ai_manager.get() != nullptr)
        ai_manager->start();
        else
        std::cout << "PhysicsManager::start(): ai_manager = nullptr. Skipping AIManager::start() call\n";
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
        if (ai_manager.get() != nullptr)
            ai_manager->stop();
        engine->shutdown();
        while (!creaturesQueue.empty()) {
            creaturesQueue.pop();
        }
        if (generationManager.get())
            generationManager.reset();
        std::cout<<"Physics engine has been shut down\n";
    }
}

void PhysicsManager::run() {
    using namespace std::chrono;
    
    // Constants for fixed timestep and target frame rate
    constexpr milliseconds TARGET_FRAME_TIME(16); // 16 ms is ~60 FPS (1000ms/60 ≈ 16.66ms)
    auto previous_time = high_resolution_clock::now();

    std::chrono::_V2::system_clock::time_point current_time;
    std::chrono::nanoseconds elapsed_time;

    tickCounter = 0;
    const int AI_UPDATE_INTERVAL = 10;

    while (running.load()) {
        current_time = high_resolution_clock::now();
        elapsed_time = current_time - previous_time;
        previous_time = current_time;
        
        if (ai_manager && tickCounter % AI_UPDATE_INTERVAL == 0) {
            std::vector<CreaturePhysicsInputs> data;

            engine->getCreatureAIInputs(data);
            ai_manager->requestCalculation(data);
        }
        if (ai_manager && ai_manager->isCalculationCompleted()) {
            auto results = ai_manager->getResults();
            BrainEditor::updateMemory(results);
            engine->applyAIResults(results);
        }
        generationManager->onTick(tickCounter);
        #if USE_HOOKS
        checkHooks(tickCounter);
        #endif

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

        const auto sleep_time = TARGET_FRAME_TIME - processing_time;
        // std::cout << sleep_time.count() << " ms" << std::endl;
        if (sleep_time > 0ms) {
            std::this_thread::sleep_for(sleep_time);
        }

        tickCounter++;
    }
}

#if USE_HOOKS
void PhysicsManager::addHook(tickHook& newHook)
{
    std::lock_guard<std::mutex> lock(hooksMutex);
    newHook.targetTick += tickCounter;
    hooks.insert(newHook);
}

void PhysicsManager::checkHooks(unsigned long currentTick)
{
    std::unique_lock<std::mutex> lock(hooksMutex);

    while (!hooks.empty() && hooks.begin()->targetTick == currentTick) {
        auto range = hooks.equal_range(*hooks.begin());

        for (auto it = range.first; it != range.second; ++it) {
            if (it->cv) {
                it->cv->notify_one();
            }
        }

        hooks.erase(range.first, range.second);
    }
}
#endif
