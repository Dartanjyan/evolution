#include "GenerationManager.h"
#include "PhysicsManager.h"
#include "BrainMutator.h"

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

const unsigned LEAVE_OLD_CREATURES = 1;
void GenerationManager::endGeneration()
{
    std::cout << "=== End of generation " << generation << " ===" << std::endl;
    
    std::vector<Creature *> creatures;
    physicsManager->getCreatures(creatures);
    std::sort(creatures.begin(), creatures.end(), [] (Creature* a, Creature* b) { return a->getFitness() > b->getFitness(); });

    // std::cout << "Cut creatures from "<<creatures.size()<<" to "<<creatures.size()/2<<"\n";
    // creatures.resize(creatures.size() / 2);
    
    Creature* bestCreature = creatures[0];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(creatures.begin(), creatures.end()-(creatures.size()/2), gen);
    for (unsigned i=0; i < creatures.size()/2; i++) {
        Creature* c = creatures[i];
        std::cout << "Creature " << c->getId() << ": fitness=" << c->getFitness() << "\n";
    }
    
    BrainMutator mutator(0.1, 0.1);
    std::vector<Brain *> newGenerationBrains {new Brain(*(bestCreature->getBrain()))};
    for (unsigned i=0; i < creaturesPerGeneration-LEAVE_OLD_CREATURES; i++) {
        Brain* childBrain = mutator.createChildBrain(
            *(creatures[i]->getBrain()), 
            *(creatures[i+1]->getBrain()),
            BrainMutator::CrossoverMethod::UNIFORM_50_50,
            BrainMutator::MutationMethod::CHANCE_FOR_EVERY_WEIGHT
        );
        newGenerationBrains.emplace_back(childBrain);
    }

    // std::cout << "New gen size is now " << newGenerationBrains.size() << " :)\n";

    physicsManager->removeAllCreatures();
    for (auto b : newGenerationBrains) {
        physicsManager->addCreature(Creature::createBasicCreature(b));
    }
    
    generation++;
}
