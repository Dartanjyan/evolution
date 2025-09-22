#include "GenerationManager.h"
#include "PhysicsManager.h"

GenerationManager::GenerationManager(PhysicsManager *physicsManager, unsigned creaturesPerGeneration, unsigned long ticksPerGeneration)
    : physicsManager(physicsManager), creaturesPerGeneration(creaturesPerGeneration), ticksPerGeneration(ticksPerGeneration)
{
    for (unsigned int i=0; i<creaturesPerGeneration; ++i) {
        physicsManager->addCreature(Creature::createBasicCreature());
    }
    std::cout << "Created GenerationManager\n";
}

GenerationManager::~GenerationManager()
{
    std::cout << "Deleted GeneraionManager\n";
}

void GenerationManager::onTick(unsigned long tick)
{
    if (tick > 0 && tick % ticksPerGeneration == 0) {
        endGeneration();
    }
}

void GenerationManager::endGeneration()
{
    std::cout << "=== End of generation " << generation << " ===" << std::endl;
    
    // TODO: запросить список существ из physicsManager
    // auto creatures = physicsManager->getCreatures();

    // TODO: отсортировать по fitness
    // std::sort(creatures.begin(), creatures.end(), ...);
    
    // TODO: оставить лучших, остальных удалить
    // TODO: создать новых детей и добавить через physicsManager->addCreature()
    
    generation++;
}
