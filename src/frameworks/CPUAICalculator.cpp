#include <vector>
#include <random>
#include "CPUAICalculator.h"
#include <iostream>
#include <Eigen/Dense>

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
    // std::cout << "CPUAICalculator::calculate() called!\n";
    // TODO: This is a placeholder that returns random numbers

    for (auto &d : data) {
        const Brain* brain = d.creature->getBrain();
        auto weights = brain->getWeights();
        auto biases = brain->getBiases();
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& d : data) {
        int num_muscles = d.creature->getMuscles().size();
        d.outputs.resize(num_muscles);
        
        for (int i = 0; i < num_muscles; i++) {
            d.outputs[i] = dist(gen);
        }
    }
}
