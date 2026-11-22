// BrainMutator.h
#ifndef BRAIN_MUTATOR_H
#define BRAIN_MUTATOR_H

#include "Brain.h"
#include <random>

struct MutatorConfig {
    float crossoverChance;
    float mutationChance;
    float mutationStrength;
};

class BrainMutator {
public:
    enum CrossoverMethod {
        UNIFORM_50_50,      // 50% от одного родителя, 50% от второго
        PROPORTIONAL,       // случайный процент (50%-90%) от первого родителя
        NO_CROSSOVER,       // мозг остаётся без изменений
        SLICE_REPLACE       // срез весов заменяется весами другого родителя
    };

    enum MutationMethod {
        CHANCE_FOR_EVERY_WEIGHT,     // каждый вес имеет шанс мутировать
        PERCENTAGE_WEIGHTS  // опр. процент весов мутирует
    };

    // mutation_rate is percents
    // mutation_strength is a random distribution parameter
    BrainMutator(MutatorConfig config);
    
    // Основной метод для создания нового мозга
    Brain* createChildBrain(const Brain& parent1, const Brain& parent2, 
                           CrossoverMethod crossover_method, 
                           MutationMethod mutation_method);

private:
    MutatorConfig config;
    std::mt19937 rng_;

    // Алгоритмы скрещивания
    Brain* uniformCrossover(const Brain& parent1, const Brain& parent2);
    Brain* proportionalCrossover(const Brain& parent1, const Brain& parent2);
    Brain* noCrossover(const Brain& parent1, const Brain& parent2);
    Brain* sliceReplaceCrossover(const Brain& parent1, const Brain& parent2);
    
    // Алгоритмы мутации
    void mutateRandomWeights(Brain& brain);
    void mutatePercentageWeights(Brain& brain);
    
    // Вспомогательные методы
    void copyWeightsAndBiases(const Brain& source, Brain& target);
    // bool validateBrainStructure(const Brain& brain1, const Brain& brain2);
};

#endif // BRAIN_MUTATOR_H