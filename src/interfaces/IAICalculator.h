#ifndef IAICALCULATOR_H
#define IAICALCULATOR_H

#include "CreaturePhysicsInputs.h"

class IAICalculator {
public:
    virtual ~IAICalculator() = default;
    
    virtual void initialize() = 0;
    virtual void shutdown() = 0;
    virtual void calculate(std::vector<CreaturePhysicsInputs>& data) = 0;
};

#endif