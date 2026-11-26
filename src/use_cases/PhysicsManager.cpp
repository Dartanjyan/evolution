#include "PhysicsManager.h"
#include <chrono>
#include <stdexcept>
#include "AIManager.h"
#include "BrainEditor.h"
#include "DrawCommandCollector.h"

PhysicsManager::PhysicsManager(std::unique_ptr<IPhysicsEngine> engine, std::unique_ptr<IAICalculator> ai_calculator, std::unique_ptr<ISimulationSaver> simulationSaver)
    : engine(std::move(engine)), running(false), simulationSaver(std::move(simulationSaver)), tickCounter(0), updateInterval(std::chrono::milliseconds(16))
{
    ai_manager = std::make_unique<AIManager>(std::move(ai_calculator));
}

void PhysicsManager::getRenderObjects(std::vector<BodyObject> &bodies, std::vector<ShapeObject> &shapes, std::vector<ConstraintObject> &constraints) const
{
    if(running.load()) {
        engine->getRenderObjects(bodies, shapes, constraints);
    }
}

void PhysicsManager::removeCreature(Creature *creature) { engine->removeCreature(creature->getId()); }

void PhysicsManager::getCreatures(std::vector<Creature *> &out) const {
    // NOTE: Might be not thread safe..?
    engine->getCreatures(out);
}

PhysicsManager::~PhysicsManager() { stop(); }

void PhysicsManager::startInternal() {
    if (ai_manager.get() != nullptr)
        ai_manager->start();
    else
        std::cout << "PhysicsManager::start(): ai_manager = nullptr. Skipping AIManager::start() call\n";
    engine->initialize();
}

void PhysicsManager::stopInternal() {
    if (ai_manager.get() != nullptr)
        ai_manager->stop();
    aiRequestPending = false; // AIManager::stop() has dropped the request together with its data
    engine->shutdown();
}

void PhysicsManager::start(DrawCommandCollector* drawCommandCollector) {
    if (running.load()) {
        std::cout << "Physics thread already running\n";
        return;
    }
    generationManager = std::make_unique<GenerationManager>(this, drawCommandCollector, 20, 60*10);
    startInternal();
    tickCounter = 0;
    startThread();
}

void PhysicsManager::startThread() {
    // `running` has to be true BEFORE the thread starts: run() begins with `while (running.load())`,
    // so setting the flag after std::thread() lets the new thread see `false` and exit immediately
    // (the simulation then never ticks: no creatures appear, nothing evolves).
    running.store(true);
    physicsThread = std::thread(&PhysicsManager::run, this);
    // std::cout << "Created new physics thread\n";
}

void PhysicsManager::loadSimulation() {
    if (!running.load())
        throw std::logic_error("The simulation is not running");

    running.store(false);
    physicsThread.join();

    LoadedSimulation save;
    try {
        save = simulationSaver->loadSimulation();
    } catch (...) {
        startThread();
        throw;
    }

    clearCreaturesQueue();
    stopInternal();
    startInternal();
    for (auto& creature : save.creatures) {
        engine->addCreature(creature.get());
        creature.release();
    }
    generationManager->restoreGeneration(save.generation, save.creatures.size());
    tickCounter = 0;
    startThread();
}

void PhysicsManager::stop() {
    if (running.load()) {
        running.store(false);
        if (physicsThread.joinable())
            physicsThread.join();
        SimulationSave save;
        save.generation = generationManager->getGeneration();
        engine->getCreatures(save.creatures);
        if (!save.creatures.empty())
            simulationSaver->saveSimulation(save);
        stopInternal();
        clearCreaturesQueue();
        // std::cout<<"Physics engine has been shut down\n";
    }
    if (generationManager.get())
        generationManager.reset();
}

void PhysicsManager::clearCreaturesQueue() {
    std::lock_guard<std::mutex> lock(creaturesQueueMutex);
    while (!creaturesQueue.empty()) {
        delete creaturesQueue.front();
        creaturesQueue.pop();
    }
}

void PhysicsManager::removeAllCreatures() { 
    tickCounter = 0;
    engine->removeAllCreatures();
}

void PhysicsManager::run() {
    using namespace std::chrono;
    
    // Constants for fixed timestep and target frame rate
    auto previous_time = high_resolution_clock::now();

    std::chrono::_V2::system_clock::time_point current_time;
    std::chrono::nanoseconds elapsed_time;

    const int AI_UPDATE_INTERVAL = 10;
    std::queue<Creature*> pendingCreatures;

    while (running.load()) {
        current_time = high_resolution_clock::now();
        elapsed_time = current_time - previous_time;
        previous_time = current_time;
        
        // Apply AI results on the tick after submission, independent of worker timing.
        if (aiRequestPending) {
            aiRequestPending = false;
            auto results = ai_manager->waitForResults();
            BrainEditor::updateMemory(results);
            engine->applyAIResults(results);
        }
        generationManager->onTick(tickCounter);
        if (tickCounter % AI_UPDATE_INTERVAL == 0) {
            std::vector<CreaturePhysicsInputs> data;
            
            engine->getCreatureAIInputs(data);
            if (!data.empty())
                aiRequestPending = ai_manager->requestCalculation(data);
        }
        const float dt = 0.01f;
        engine->update(dt);

        // Handle all pending creature additions in this frame
        {
            std::lock_guard<std::mutex> lock(creaturesQueueMutex);
            pendingCreatures.swap(creaturesQueue);
        }
        while (!pendingCreatures.empty()) {
            engine->addCreature(pendingCreatures.front());
            pendingCreatures.pop();
        }

        // Update rate regulation
        const auto processing_time = duration_cast<milliseconds>(
            high_resolution_clock::now() - current_time
        );
        const auto sleep_time = this->updateInterval - processing_time;
        // std::cout << sleep_time.count() << " ms" << std::endl;
        if (sleep_time > 0ms) {
            std::this_thread::sleep_for(sleep_time);
        }

        tickCounter++;
    }
}

void PhysicsManager::addCreature(Creature* creature)
{
    std::lock_guard<std::mutex> lock(creaturesQueueMutex);
    creaturesQueue.push(creature);
}

void PhysicsManager::setUpdateTimeScale(float scale)
{
    if (scale == 0) {
        updateInterval = std::chrono::milliseconds(0);
    } else {
        updateInterval = std::chrono::milliseconds((int)(16.0f/scale));
    }

    std::cout << "Time scale: " << scale << ", interval = " << updateInterval << "\n";
}

unsigned int PhysicsManager::getGeneration() const {
    return generationManager ? generationManager->getGeneration() : 0;
}
