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
    if (tick % ticksPerGeneration == 0 && tick != 0) {
        endGeneration();
    }
}

const unsigned LEAVE_OLD_CREATURES = 1;
const unsigned NEW_RANDOM_CREATURES = 2;
void GenerationManager::endGeneration()
{
    std::cout << "=== End of generation " << generation << " ===" << std::endl;
    
    std::vector<Creature *> creatures;
    physicsManager->getCreatures(creatures);
    if (creatures.size() == 0) {
        std::cout << "Got 0 creatures at the end of generation. Skipping.\n";
        return;
    }
    std::sort(creatures.begin(), creatures.end(), [] (Creature* a, Creature* b) { return a->getFitness() > b->getFitness(); });

    // std::cout << "Cut creatures from "<<creatures.size()<<" to "<<creatures.size()/2<<"\n";
    // creatures.resize(creatures.size() / 2);
    for (unsigned i=0; i < creatures.size()/2; i++) {
        Creature* c = creatures[i];
        std::cout << "Creature " << c->getId() << ": fitness=" << c->getFitness() << "\n";
    }
    
    Creature* bestCreature = creatures[0];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(creatures.begin(), creatures.end()-(creatures.size()/2), gen);
    
    // 70% of better parent's genes
    // Every weight has 5% of chance to be mutated
    // 0.1 mutation random distribution
    struct MutatorConfig config;
    config.crossoverChance = 0.7;
    config.mutationChance = 0.05;
    config.mutationStrength = 0.1;
    
    BrainMutator mutator(config);
    std::vector<Brain *> newGenerationBrains {new Brain(*(bestCreature->getBrain()))};

    unsigned int newChildren = creaturesPerGeneration;
    newChildren -= (newChildren >= LEAVE_OLD_CREATURES ? LEAVE_OLD_CREATURES : 0);
    newChildren -= (newChildren >= NEW_RANDOM_CREATURES ? NEW_RANDOM_CREATURES : 0);
    for (unsigned i=0; i < newChildren; i++) {
        Brain* childBrain = mutator.createChildBrain(
            *(creatures[i]->getBrain()), 
            *(creatures[newChildren > 1 ? i+1 : i]->getBrain()),
            BrainMutator::CrossoverMethod::PROPORTIONAL,
            BrainMutator::MutationMethod::CHANCE_FOR_EVERY_WEIGHT
        );
        newGenerationBrains.emplace_back(childBrain);
    }
    
    // std::cout << "New gen size is now " << newGenerationBrains.size() << " :)\n";
    
    // physicsManager->removeAllCreatures();
    // std::cout << "Rebooting physicsManager\n";
    physicsManager->stopInternal();
    physicsManager->startInternal();
    for (auto b : newGenerationBrains) {
        physicsManager->addCreature(Creature::createBasicCreature(b));
    }
    if ((creaturesPerGeneration - newChildren - LEAVE_OLD_CREATURES) > 0) {
        for (unsigned i=0; i < NEW_RANDOM_CREATURES; i++) {
            physicsManager->addCreature(Creature::createBasicCreature());
        }
    }
    
    generation++;
}
