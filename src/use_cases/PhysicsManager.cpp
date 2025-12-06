#include "PhysicsManager.h"
#include <chrono>
#include "AIManager.h"
#include "BrainEditor.h"

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

void PhysicsManager::getCreatures(std::vector<Creature *> &out) {
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
    engine->shutdown();
    while (!creaturesQueue.empty()) {
        creaturesQueue.pop();
    }
}

void PhysicsManager::start() {
    generationManager = std::make_unique<GenerationManager>(this, 20, 600);
    if (!running.load()) {
        startInternal();
        physicsThread = std::thread(&PhysicsManager::run, this);
        running.store(true);
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
        SimulationSave save;
        save.generation = generationManager->getGeneration();
        engine->getCreatures(save.creatures);
        simulationSaver->saveSimulation(save);
        stopInternal();
        std::cout<<"Physics engine has been shut down\n";
    }
    if (generationManager.get())
        generationManager.reset();
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

        const auto sleep_time = this->updateInterval - processing_time;
        // std::cout << sleep_time.count() << " ms" << std::endl;
        if (sleep_time > 0ms) {
            std::this_thread::sleep_for(sleep_time);
        }

        tickCounter++;
    }
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

unsigned int PhysicsManager::getGeneration() {
    return generationManager ? generationManager->getGeneration() : 0;
}
