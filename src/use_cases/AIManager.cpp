#include "AIManager.h"

AIManager::AIManager(std::unique_ptr<IAICalculator> calculator)
    : calculator(std::move(calculator)), running(false)
{
}

AIManager::~AIManager()
{
    stop();
}

void AIManager::start() {
    if (!running.load()) {
        running.store(true);
        calculator->initialize();
        aiThread = std::thread(&AIManager::run, this);
        // std::cout << "Created new AI thread\n";
    } else {
        std::cout << "AI thread already running\n";
    }
}

void AIManager::stop() {
    if (running.load()) {
        running.store(false);
        cv.notify_one(); // awoke thread to end it
        if (aiThread.joinable())
            aiThread.join();
        calculator->shutdown();
        std::lock_guard<std::mutex> lock(mtx);
        currentData.clear();
        calculationRequested = false;
        calculationCompleted = false;
        // std::cout<<"AI engine has been shut down\n";
    }
}

void AIManager::requestCalculation(std::vector<CreaturePhysicsInputs>& data) {
    std::lock_guard<std::mutex> lock(mtx);
    currentData = std::move(data);
    calculationRequested = true;
    calculationCompleted = false;
    cv.notify_one();
}

bool AIManager::isCalculationCompleted() {
    return calculationCompleted.load();
}

void AIManager::run() {
    while (running.load()) {
        std::unique_lock<std::mutex> lock(mtx);
        
        // Sleep until calculation is requested (or app quit)
        cv.wait(lock, [this] {
            return calculationRequested.load() || !running.load();
        });
        if (!running.load()) break;
        
        calculationRequested.store(false);
        lock.unlock();
        
        calculator->calculate(currentData);
        
        calculationCompleted.store(true);
    }
    // std::cout << "AI thread exit\n";
}

const std::vector<CreaturePhysicsInputs>& AIManager::getResults() {
    calculationCompleted.store(false);
    return currentData;
}
