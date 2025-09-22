#include "GenerationManager.h"
#include "PhysicsManager.h"

GenerationManager::GenerationManager(PhysicsManager *physicsManager, unsigned creaturesPerGeneration, unsigned long ticksPerGeneration)
    : physicsManager(physicsManager), creaturesPerGeneration(creaturesPerGeneration), ticksPerGeneration(ticksPerGeneration), generation(0)
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
    
    std::vector<Creature *> creatures;
    physicsManager->getCreatures(creatures);
    std::sort(creatures.begin(), creatures.end(), [] (Creature* a, Creature* b) { return a->getFitness() > b->getFitness(); });

    for (auto c : creatures) {
        std::cout << "Creature " << c->getId() << ": fitness=" << c->getFitness() << "\n";
    }

    // TODO: запросить список существ из physicsManager
    // auto creatures = physicsManager->getCreatures();

    // TODO: отсортировать по fitness
    // std::sort(creatures.begin(), creatures.end(), ...);
    
    // TODO: оставить лучших, остальных удалить
    // TODO: создать новых детей и добавить через physicsManager->addCreature()
    
    generation++;
}
