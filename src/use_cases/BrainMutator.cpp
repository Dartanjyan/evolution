#include "BrainMutator.h"
#include <algorithm>
/*
BrainMutator::BrainMutator(double mut_rate, double mut_strength)
  : mutation_rate_(mut_rate),
  mutation_strength_(mut_strength),
  rng_(std::random_device{}())
{}

void BrainMutator::mutate(Brain& brain) {
    std::vector<double> genome = brain.encodeGenome();
    std::uniform_real_distribution<double> coin(0.0, 1.0), dist(-mutation_strength_, mutation_strength_);

    for (double& gene : genome) {
        if (coin(rng_) < mutation_rate_) {
            gene += dist(rng_);
        }
    }
    brain.decodeGenome(genome);
}

// Returns a new Brain - a crossover of the two given Brains
Brain* BrainMutator::crossover(const Brain& p1, const Brain& p2) {
    std::vector<double> g1 = p1.encodeGenome();
    std::vector<double> g2 = p2.encodeGenome();
    std::vector<double> child_genome(g1.size());

    std::uniform_int_distribution<size_t> split(0, g1.size());
    size_t point = split(rng_);

    for (size_t i = 0; i < g1.size(); ++i) {
        child_genome[i] = (i < point ? g1[i] : g2[i]);
    }

    Brain* child = new Brain(p1.getLayerSizes());
    child->decodeGenome(child_genome);
    return child;
}
*/
