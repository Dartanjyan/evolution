#include "GenerationManager.h"
#include "PhysicsManager.h"
#include "BrainMutator.h"
#include "DrawCommandCollector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

GenerationManager::GenerationManager(PhysicsManager *physicsManager, DrawCommandCollector* drawCommandCollector, unsigned creaturesPerGeneration, unsigned long ticksPerGeneration)
    : physicsManager(physicsManager), drawCommandCollector(drawCommandCollector), creaturesPerGeneration(creaturesPerGeneration), ticksPerGeneration(ticksPerGeneration), generation(0)
{
    for (unsigned int i=0; i<creaturesPerGeneration; ++i) {
        physicsManager->addCreature(Creature::createBasicCreature());
    }

    updateScreenInfo();

    std::cout << "Created GenerationManager\n";
}

void GenerationManager::updateScreenInfo() {
    #if DRAW_NEURAL_NETWORK
    if (!physicsManager || !drawCommandCollector)
        return;

    std::vector<Creature *> creatures;
    physicsManager->getCreatures(creatures);
    drawCommandCollector->setCreatures(creatures);
    drawCommandCollector->updateScreenInfo();
    #endif
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

void GenerationManager::restoreGeneration(unsigned number, unsigned populationSize)
{
    generation.store(number);
    creaturesPerGeneration = populationSize;
    updateScreenInfo();
}

// How many of the best creatures go to the next generation as exact copies (elitism)
const unsigned LEAVE_OLD_CREATURES = 1;
// How many brand-new random creatures are added every generation (diversity)
const unsigned NEW_RANDOM_CREATURES = 2;

namespace {
// Fitness used for sorting. NaN (e.g. the physics blew up) would break the strict weak ordering
// std::sort needs, so such a creature is simply treated as the worst one possible.
float sortableFitness(const Creature* c)
{
    const float f = c->getFitness();
    return std::isnan(f) ? -std::numeric_limits<float>::infinity() : f;
}
}

void GenerationManager::endGeneration()
{
    updateScreenInfo();
    
    std::cout << "=== End of generation " << generation << " ===" << std::endl;
    
    std::vector<Creature *> creatures;
    physicsManager->getCreatures(creatures);
    if (creatures.size() == 0) {
        std::cout << "Got 0 creatures at the end of generation. Skipping.\n";
        return;
    }
    // Best first. stable_sort keeps the order reproducible for equal fitness values.
    std::stable_sort(creatures.begin(), creatures.end(), [] (const Creature* a, const Creature* b) {
        return sortableFitness(a) > sortableFitness(b);
    });

    for (unsigned i=0; i < creatures.size()/2; i++) {
        Creature* c = creatures[i];
        std::cout << "Creature " << c->getId() << ": fitness=" << c->getFitness() << "\n";
    }
    {
        double sum = 0;
        unsigned finite = 0;
        for (const Creature* c : creatures) {
            if (std::isfinite(c->getFitness())) {
                sum += c->getFitness();
                ++finite;
            }
        }
        std::cout << "Generation " << generation << " fitness: best=" << creatures.front()->getFitness()
                  << " median=" << creatures[creatures.size()/2]->getFitness()
                  << " worst=" << creatures.back()->getFitness()
                  << " mean=" << (finite ? sum / finite : 0.0) << std::endl;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    struct MutatorConfig config;
    config.crossoverChance = 0.8;  // percentage of better parent's genes
    config.mutationChance = 0.05;  // chance for every weight to be mutated
    config.mutationStrength = 0.05; // standard deviation of the change of a mutated weight
    
    BrainMutator mutator(config);

    // How many creatures of each kind the new generation gets
    const unsigned eliteCount = std::min<size_t>({LEAVE_OLD_CREATURES, creaturesPerGeneration, creatures.size()});
    // random newcomers only if they do not crowd out all the children
    const unsigned newcomers = creaturesPerGeneration > eliteCount + NEW_RANDOM_CREATURES ? NEW_RANDOM_CREATURES : 0;
    const unsigned newChildren = creaturesPerGeneration - eliteCount - newcomers;

    const auto& bodyTemplate = *creatures.front();
    std::vector<std::unique_ptr<Creature>> newGeneration;
    newGeneration.reserve(creaturesPerGeneration);

    for (unsigned i=0; i < eliteCount; i++) {
        newGeneration.push_back(bodyTemplate.cloneWithBrain(std::make_unique<Brain>(*creatures[i]->getBrain())));
    }

    // Selection: only the better half of the population breeds (truncation selection).
    // `creatures` is sorted best-first, so the breeding pool is simply its first poolSize elements.
    // (Before, the parents were taken from the whole sorted list, so a third of them came from the
    // worse half and the best creature had no more children than the ones just below the median.)
    const size_t poolSize = std::max<size_t>(std::min<size_t>(2, creatures.size()), creatures.size() / 2);
    std::uniform_int_distribution<size_t> pickParent(0, poolSize - 1);

    for (unsigned i=0; i < newChildren; i++) {
        size_t a = pickParent(gen);
        size_t b = pickParent(gen);
        if (poolSize > 1) {
            while (b == a)
                b = pickParent(gen);  // two different parents
        }
        // The first parent gives the larger share of genes (see CrossoverMethod::PROPORTIONAL),
        // so it must be the better one, i.e. the one with the smaller index.
        if (b < a)
            std::swap(a, b);
        std::unique_ptr<Brain> childBrain(mutator.createChildBrain(
            *(creatures[a]->getBrain()),
            *(creatures[b]->getBrain()),
            BrainMutator::CrossoverMethod::PROPORTIONAL,
            BrainMutator::MutationMethod::CHANCE_FOR_EVERY_WEIGHT
        ));
        newGeneration.push_back(bodyTemplate.cloneWithBrain(std::move(childBrain)));
    }
    for (unsigned i=0; i < newcomers; i++) {
        const auto* brain = bodyTemplate.getBrain();
        newGeneration.push_back(bodyTemplate.cloneWithBrain(
            std::make_unique<Brain>(brain->getLayerSizes(), brain->getMemory().size())));
    }
    
    physicsManager->stopInternal();
    physicsManager->startInternal();
    for (auto& creature : newGeneration) {
        physicsManager->addCreature(creature.get());
        creature.release();
    }
    
    generation++;
}
