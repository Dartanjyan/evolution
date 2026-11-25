#ifndef AIMANAGER_H
#define AIMANAGER_H

#include <memory>
#include <thread>
#include <atomic>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "CreaturePhysicsInputs.h"
#include "IAICalculator.h"

class AIManager {
private:
    void run();
    std::unique_ptr<IAICalculator> calculator;
    std::thread aiThread;
    std::atomic<bool> running;
    
    // Synchronization stuff
    std::mutex mtx;
    std::condition_variable cv;        // wakes the AI thread (new request / stop)
    std::condition_variable doneCv;    // wakes the caller waiting for results
    std::atomic<bool> calculationRequested{false};
    std::atomic<bool> calculationCompleted{false};
    // Prevents a new request from replacing currentData before its results are taken.
    bool busy = false;

    std::vector<CreaturePhysicsInputs> currentData;
public:
    AIManager(std::unique_ptr<IAICalculator> calculator);
    ~AIManager();

    void start();
    void stop();
    // Calculation request from PhysicsManager.
    // Returns false (and leaves `data` untouched) if the previous request has not been consumed yet.
    bool requestCalculation(std::vector<CreaturePhysicsInputs>& data);
    // Waits for and takes the results of the pending request.
    std::vector<CreaturePhysicsInputs> waitForResults();
};

#endif
