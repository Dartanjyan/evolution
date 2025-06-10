#include "AIManager.h"

AIManager::AIManager(std::unique_ptr<IAICalculator> calculator)
    : calculator(std::move(calculator))
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
        std::cout << "Created new AI thread\n";
    } else {
        std::cout << "AI thread already running\n";
    }
}

void AIManager::stop() {
    if (running.load()) {
        running.store(false);
        if (aiThread.joinable())
            aiThread.join();
        calculator->shutdown();
        std::cout<<"AI engine has been shut down\n";
    }
}

void AIManager::run() {
    std::cout << "AIManager::run() is now placeholder!\n";
}