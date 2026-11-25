#include "Brain.h"
#include <algorithm>
#include <stdexcept> // runtime_error
#include <random>
#include <iostream>

// static unsigned int last_id
unsigned Brain::last_id = 0;
bool Brain::is_config_printed = false;

Brain::Brain() : 
    id(Brain::newId()),
    layer_sizes_({1}), 
    weights_({}), 
    biases_({}),
    memory_({})
{}

// layer_sizes is {input, h1, h2, ..., output}
//
// Each vector in weights is plained matrix of weights: weights[i] holds layer_sizes[i] * layer_sizes[i + 1] values.
// weights.size() must be equal to layer_sizes.size() - 1.
// biases holds one value per layer (without the input one), so biases.size() must be equal to layer_sizes.size() - 1 too.
//

Brain::Brain(const std::vector<std::size_t>& layer_sizes,
             const std::size_t memory,
             const std::vector<std::vector<double>>& weights,
             const std::vector<double>& biases
             )
  : id(Brain::newId()), layer_sizes_(layer_sizes), weights_(weights), biases_(biases), memory_(std::vector<double>(memory, 0))
{
    if (layer_sizes_.empty()) {
        // layer_sizes_.size() - 1 below would wrap around to a huge number
        throw std::runtime_error("Brain must have at least one layer");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 0.5);

    size_t total_weights = 0;
    for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        total_weights += layer_sizes_[i] * layer_sizes_[i + 1] + 1;
    }

    if (!Brain::is_config_printed){
        std::cout << "Brain::Brain(): layers: {";
        for (auto layer : layer_sizes_) {
            std::cout << layer << ", ";
        }
        std::cout << "\b\b}, total_weights = " << total_weights << " (" << sizeof(double) * total_weights << " bytes)\n";
        Brain::is_config_printed = true;
    }

    // If weights and biases are not given then generate them
    if (weights_.empty() && biases_.empty()) {
        for (size_t i = 0; i < layer_sizes_.size()-1; i++) {
            weights_.push_back(std::vector<double>(layer_sizes_[i] * layer_sizes_[i + 1]));
            for (auto& w : weights_[i]) {
                w = dist(gen);
            }
        }

        biases_.resize(layer_sizes_.size() - 1);
        for (auto& b : biases_) {
            // Biases are often less than weights
            b = dist(gen) * 0.5;
        }
    }
    // If weights and biases are given then check if they match layer sizes
    else {
        // (The old check compared the total number of weights with weights_.size(), which is only the
        // number of layers, so it threw for ANY given weights - e.g. when a saved brain is loaded.)
        bool valid = weights_.size() == layer_sizes_.size() - 1 &&
                     biases_.size()  == layer_sizes_.size() - 1;
        for (size_t i = 0; valid && i < weights_.size(); ++i) {
            valid = weights_[i].size() == layer_sizes_[i] * layer_sizes_[i + 1];
        }
        if (!valid) {
            throw std::runtime_error("Invalid weights/biases dimensions");
        }
    }
}

Brain::Brain(const Brain& other)
    : id(Brain::newId()),
      layer_sizes_(other.layer_sizes_),
      weights_(other.weights_),
      biases_(other.biases_),
      memory_(other.memory_.size(), 0.0)
{
}

const std::vector<size_t> Brain::getLayerSizes() const noexcept { return layer_sizes_; }
const std::vector<std::vector<double>> Brain::getWeights() const noexcept { return weights_; }
const std::vector<double> Brain::getBiases() const noexcept { return biases_; }
const std::vector<double> Brain::getMemory() const noexcept { return memory_; }
void Brain::setWeights(const std::vector<std::vector<double>>& new_weights) { weights_ = new_weights; }
void Brain::setBiases (const std::vector<double>& new_biases) { biases_  = new_biases; }
void Brain::setMemory(const std::vector<double> &new_memory) { memory_ = new_memory; }

unsigned Brain::newId() { return ++Brain::last_id; }
void Brain::resetId() { Brain::last_id = 0; }
