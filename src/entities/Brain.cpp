#include "Brain.h"
#include <algorithm>
#include <stdexcept> // runtime_error
#include <random>

unsigned Brain::last_id = 0;

Brain::Brain() : 
    id(Brain::newId()),
    layer_sizes_({1}), 
    weights_({}), 
    biases_({})
{}

Brain::Brain(const std::vector<size_t>& layer_sizes, 
             const std::vector<double>& weights,
             const std::vector<double>& biases)
  : id(Brain::newId()), layer_sizes_(layer_sizes), weights_(weights), biases_(biases)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 0.1); // mean=0, stddev=0.1

    size_t total_weights = 0;
    for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        total_weights += layer_sizes_[i] * layer_sizes_[i + 1];
    }

    // If weights and biases are not given then generate them
    if (weights_.empty() && biases_.empty()) {
        weights_.resize(total_weights);
        for (auto& w : weights_) {
            w = dist(gen);
        }

        biases_.resize(layer_sizes_.size() - 1);
        for (auto& b : biases_) {
            // Biases are often less than weights
            b = dist(gen) * 0.5;
        }
    }
    // If weights and biases are given then check if they match layer sizes
    else if (total_weights != weights_.size() || 
             biases_.size() != (layer_sizes_.size() - 1)) {
        throw std::runtime_error("Invalid weights/biases dimensions");
    }
}

// Get "Genome" of this Brain
std::vector<double> Brain::encodeGenome() const {
    std::vector<double> genome;
    genome.reserve(weights_.size() + biases_.size());
    genome.insert(genome.end(), weights_.begin(), weights_.end());
    genome.insert(genome.end(), biases_ .begin(), biases_ .end());
    return genome;
}

// Decode "Genome" and set weights and biases
void Brain::decodeGenome(const std::vector<double>& genome) {
    const size_t w = weights_.size();
    // Assume, that genome.size()==w + biases_.size()
    std::copy_n(genome.begin(), w,           weights_.begin());
    std::copy_n(genome.begin() + w, biases_.size(), biases_.begin());
}

const std::vector<size_t>& Brain::getLayerSizes() const noexcept { return layer_sizes_; }
const std::vector<double>& Brain::getWeights() const noexcept { return weights_; }
const std::vector<double>& Brain::getBiases() const noexcept { return biases_; }
void Brain::setWeights(const std::vector<double>& w) { weights_ = w; }
void Brain::setBiases (const std::vector<double>& b) { biases_  = b; }

unsigned Brain::newId() { return ++Brain::last_id; }
void Brain::resetId() { Brain::last_id = 0; }
