// BrainMutator.cpp
#include "BrainMutator.h"
#include <algorithm>
#include <iostream>

BrainMutator::BrainMutator(MutatorConfig config)
    : config(config),
      rng_(std::random_device{}())
{}

Brain* BrainMutator::createChildBrain(const Brain& parent1, const Brain& parent2, 
                                     CrossoverMethod crossover_method, 
                                     MutationMethod mutation_method) {
    
    // if (!validateBrainStructure(parent1, parent2)) {
    //     std::cerr << "Brain structures don't match! Using parent1 as fallback.\n";
    //     return new Brain(parent1);
    // }

    Brain* child_brain = nullptr;
    
    // Применяем выбранный алгоритм скрещивания
    switch (crossover_method) {
        case UNIFORM_50_50:
            child_brain = uniformCrossover(parent1, parent2);
            break;
        case PROPORTIONAL:
            child_brain = proportionalCrossover(parent1, parent2);
            break;
        case NO_CROSSOVER:
            child_brain = noCrossover(parent1, parent2);
            break;
        case SLICE_REPLACE:
            child_brain = sliceReplaceCrossover(parent1, parent2);
            break;
    }
    
    // Применяем выбранный алгоритм мутации
    if (child_brain) {
        switch (mutation_method) {
            case CHANCE_FOR_EVERY_WEIGHT:
                mutateRandomWeights(*child_brain);
                break;
            case PERCENTAGE_WEIGHTS:
                mutatePercentageWeights(*child_brain);
                break;
        }
    }
    
    return child_brain;
}

void BrainMutator::copyWeightsAndBiases(const Brain& source, Brain& target) {
    target.setWeights(source.getWeights());
    target.setBiases(source.getBiases());
}

// 1. Скрещивание 50/50
Brain* BrainMutator::uniformCrossover(const Brain& parent1, const Brain& parent2) {
    Brain* child = new Brain(parent1.getLayerSizes(), parent1.getMemory().size());
    
    auto weights1 = parent1.getWeights();
    auto weights2 = parent2.getWeights();
    auto biases1 = parent1.getBiases();
    auto biases2 = parent2.getBiases();
    
    std::vector<std::vector<double>> child_weights;
    std::vector<double> child_biases;
    
    std::bernoulli_distribution dist(0.5);
    
    // Скрещивание весов
    for (size_t layer = 0; layer < weights1.size(); ++layer) {
        std::vector<double> layer_weights;
        for (size_t i = 0; i < weights1[layer].size(); ++i) {
            if (dist(rng_)) {
                layer_weights.push_back(weights1[layer][i]);
            } else {
                layer_weights.push_back(weights2[layer][i]);
            }
        }
        child_weights.push_back(layer_weights);
    }
    
    // Скрещивание смещений
    for (size_t i = 0; i < biases1.size(); ++i) {
        if (dist(rng_)) {
            child_biases.push_back(biases1[i]);
        } else {
            child_biases.push_back(biases2[i]);
        }
    }
    
    child->setWeights(child_weights);
    child->setBiases(child_biases);
    
    return child;
}

// 2. Пропорциональное скрещивание (50%-90% от первого родителя)
Brain* BrainMutator::proportionalCrossover(const Brain& parent1, const Brain& parent2) {
    Brain* child = new Brain(parent1.getLayerSizes(), parent1.getMemory().size());
    
    std::uniform_real_distribution<double> percent_dist(0.5, 0.9);
    double percent_from_parent1 = percent_dist(rng_);
    
    auto weights1 = parent1.getWeights();
    auto weights2 = parent2.getWeights();
    auto biases1 = parent1.getBiases();
    auto biases2 = parent2.getBiases();
    
    std::vector<std::vector<double>> child_weights;
    std::vector<double> child_biases;
    
    std::bernoulli_distribution dist(percent_from_parent1);
    
    // Скрещивание весов
    for (size_t layer = 0; layer < weights1.size(); ++layer) {
        std::vector<double> layer_weights;
        for (size_t i = 0; i < weights1[layer].size(); ++i) {
            if (dist(rng_)) {
                layer_weights.push_back(weights1[layer][i]);
            } else {
                layer_weights.push_back(weights2[layer][i]);
            }
        }
        child_weights.push_back(layer_weights);
    }
    
    // Скрещивание смещений
    for (size_t i = 0; i < biases1.size(); ++i) {
        if (dist(rng_)) {
            child_biases.push_back(biases1[i]);
        } else {
            child_biases.push_back(biases2[i]);
        }
    }
    
    child->setWeights(child_weights);
    child->setBiases(child_biases);
    
    return child;
}

// 3. Без скрещивания (случайный родитель)
Brain* BrainMutator::noCrossover(const Brain& parent1, const Brain& parent2) {
    Brain* child = new Brain(parent1.getLayerSizes(), parent1.getMemory().size());
    
    std::bernoulli_distribution dist(0.5);
    
    if (dist(rng_)) {
        copyWeightsAndBiases(parent1, *child);
    } else {
        copyWeightsAndBiases(parent2, *child);
    }
    
    return child;
}

// 4. Замена среза весов
Brain* BrainMutator::sliceReplaceCrossover(const Brain& parent1, const Brain& parent2) {
    Brain* child = new Brain(parent1.getLayerSizes(), parent1.getMemory().size());
    
    // Копируем весь мозг от первого родителя
    copyWeightsAndBiases(parent1, *child);
    
    auto weights2 = parent2.getWeights();
    auto biases2 = parent2.getBiases();
    
    std::uniform_int_distribution<size_t> layer_dist(0, weights2.size() - 1);
    size_t target_layer = layer_dist(rng_);
    
    if (!weights2[target_layer].empty()) {
        std::uniform_int_distribution<size_t> start_dist(0, weights2[target_layer].size() - 1);
        size_t start_index = start_dist(rng_);
        
        std::uniform_int_distribution<size_t> length_dist(1, weights2[target_layer].size() - start_index);
        size_t slice_length = length_dist(rng_);
        
        // Заменяем срез весов
        auto child_weights = child->getWeights();
        for (size_t i = start_index; i < start_index + slice_length && i < weights2[target_layer].size(); ++i) {
            child_weights[target_layer][i] = weights2[target_layer][i];
        }
        child->setWeights(child_weights);
    }
    
    // Также возможна замена среза смещений
    if (!biases2.empty() && std::bernoulli_distribution(0.3)(rng_)) {
        std::uniform_int_distribution<size_t> start_dist(0, biases2.size() - 1);
        size_t start_index = start_dist(rng_);
        
        std::uniform_int_distribution<size_t> length_dist(1, biases2.size() - start_index);
        size_t slice_length = length_dist(rng_);
        
        auto child_biases = child->getBiases();
        for (size_t i = start_index; i < start_index + slice_length && i < biases2.size(); ++i) {
            child_biases[i] = biases2[i];
        }
        child->setBiases(child_biases);
    }
    
    return child;
}

// Мутация: каждый вес имеет шанс мутировать
void BrainMutator::mutateRandomWeights(Brain& brain) {
    std::uniform_real_distribution<double> chance_dist(0.0, 1.0);
    std::normal_distribution<double> change_dist(0.0, config.mutationStrength);
    
    auto weights = brain.getWeights();
    auto biases = brain.getBiases();
    

    // Мутация весов
    for (auto& layer : weights) {
        for (auto& weight : layer) {
            if (chance_dist(rng_) < config.mutationChance) {
                weight += change_dist(rng_);
            }
        }
    }
    
    // Мутация смещений
    for (auto& bias : biases) {
        if (chance_dist(rng_) < config.mutationChance) {
            bias += change_dist(rng_);
        }
    }
    
    brain.setWeights(weights);
    brain.setBiases(biases);
}

// Мутация: определенный процент весов мутирует
void BrainMutator::mutatePercentageWeights(Brain& brain) {
    std::normal_distribution<double> change_dist(0.0, config.mutationChance);
    
    auto weights = brain.getWeights();
    auto biases = brain.getBiases();
    
    // Подсчитываем общее количество параметров
    size_t total_parameters = 0;
    for (const auto& layer : weights) {
        total_parameters += layer.size();
    }
    total_parameters += biases.size();
    
    // Вычисляем количество мутирующих параметров
    size_t parameters_to_mutate = static_cast<size_t>(total_parameters * config.mutationChance);
    
    if (parameters_to_mutate == 0) return;
    
    // Создаем плоский список указателей на все параметры
    std::vector<double*> all_parameters;
    for (auto& layer : weights) {
        for (auto& weight : layer) {
            all_parameters.push_back(&weight);
        }
    }
    for (auto& bias : biases) {
        all_parameters.push_back(&bias);
    }
    
    // Перемешиваем и мутируем нужное количество параметров
    std::shuffle(all_parameters.begin(), all_parameters.end(), rng_);
    
    for (size_t i = 0; i < parameters_to_mutate && i < all_parameters.size(); ++i) {
        *(all_parameters[i]) += change_dist(rng_);
    }
    
    brain.setWeights(weights);
    brain.setBiases(biases);
}