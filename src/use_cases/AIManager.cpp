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
        {
            // `running` must be changed under the same mutex the AI thread uses in cv.wait(),
            // otherwise the notification can be lost (thread checks the predicate, then we flip the flag
            // and notify, then it goes to sleep forever and join() below never returns).
            std::lock_guard<std::mutex> lock(mtx);
            running.store(false);
        }
        cv.notify_all(); // awoke thread to end it
        doneCv.notify_all();
        if (aiThread.joinable())
            aiThread.join();
        calculator->shutdown();
        std::lock_guard<std::mutex> lock(mtx);
        currentData.clear();
        calculationRequested = false;
        calculationCompleted = false;
        busy = false;
        // std::cout<<"AI engine has been shut down\n";
    }
}

bool AIManager::requestCalculation(std::vector<CreaturePhysicsInputs>& data) {
    std::lock_guard<std::mutex> lock(mtx);
    // The AI thread works on currentData without holding the mutex, so replacing it while a
    // calculation is still running (or while its results are not yet read) would be a data race.
    if (busy || !running.load())
        return false;
    busy = true;
    currentData = std::move(data);
    calculationRequested = true;
    calculationCompleted = false;
    cv.notify_one();
    return true;
}

std::vector<CreaturePhysicsInputs> AIManager::waitForResults() {
    std::unique_lock<std::mutex> lock(mtx);
    doneCv.wait(lock, [this] {
        return calculationCompleted.load() || !busy || !running.load();
    });
    if (!running.load() || !calculationCompleted.load())
        return {};

    calculationCompleted = false;
    busy = false;
    return std::move(currentData);
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

        {
            std::lock_guard<std::mutex> doneLock(mtx);
            calculationCompleted.store(true);
        }
        doneCv.notify_all();
    }
    // std::cout << "AI thread exit\n";
}
