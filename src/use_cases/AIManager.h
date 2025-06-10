#ifndef AIMANAGER_H
#define AIMANAGER_H

#include <memory>
#include <thread>
#include <atomic>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include "IAICalculator.h"

class AIManager {
private:
    void run();
    std::unique_ptr<IAICalculator> calculator;
    std::thread aiThread;
    std::atomic<bool> running;
    
    // Synchronization stuff
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> calculationRequested{false};
    std::atomic<bool> calculationCompleted{false};

public:
    AIManager(std::unique_ptr<IAICalculator> calculator);
    ~AIManager();

    void start();
    void stop();
    void requestCalculation();  // Calculation request from PhysicsManager
    bool isCalculationCompleted(); // Check if calculations are done
};

#endif