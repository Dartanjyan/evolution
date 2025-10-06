#include <vector>
#include "CPUAICalculator.h"

CPUAICalculator::CPUAICalculator() {

}

CPUAICalculator::~CPUAICalculator() {

}

void CPUAICalculator::initialize() {
    std::cout << "CPUAICalculator::initialize() called!\n";
}

void CPUAICalculator::shutdown() {
    std::cout << "CPUAICalculator::shutdown() called!\n";
}

void CPUAICalculator::calculate(std::vector<CreaturePhysicsInputs>& data) {
    for (auto &d : data) {
        const Brain* brain = d.creature->getBrain();
        auto weights = brain->getWeights();
        auto biases = brain->getBiases();
        auto layer_sizes = brain->getLayerSizes();
        auto memory = brain->getMemory();
        
        std::vector<double> current_layer = d.inputs;
        
        // Forward pass through all layers
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            std::vector<double> next_layer(layer_sizes[layer + 1], biases[layer]);
            
            // Matrix multiplication: next = weights * current + bias
            for (size_t out_neuron = 0; out_neuron < layer_sizes[layer + 1]; ++out_neuron) {
                for (size_t in_neuron = 0; in_neuron < current_layer.size(); ++in_neuron) {
                    auto cur = current_layer[in_neuron];
                    auto w = weights[layer][out_neuron * current_layer.size() + in_neuron];  // here fail
                    next_layer[out_neuron] += cur * w;
                }
                
                // Activation function (tanh for hidden layers, sigmoid for output)
                const float COMPRESS_FUNC = 1;
                auto next_neuron = COMPRESS_FUNC * next_layer[out_neuron];

                // Hidden layers
                if (layer < weights.size() - 1) {
                    // Sigmoid
                    // next_layer[out_neuron] = 1.0 / (1.0 + std::exp(-next_neuron));

                    // RELU
                    next_layer[out_neuron] = next_neuron > 0 ? next_neuron : 0;
                }
                // Output layer
                else {
                    next_layer[out_neuron] = std::tanh(next_neuron);
                }
            }
            
            current_layer = std::move(next_layer);
        }
        
        d.outputs = current_layer;
    }
}
