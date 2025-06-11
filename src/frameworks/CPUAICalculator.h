#ifndef CPUAICALCULATOR_H
#define CPUAICALCULATOR_H

#include <iostream>
#include "IAICalculator.h"

class CPUAICalculator: public IAICalculator {
public:
    CPUAICalculator();
    ~CPUAICalculator() override;

    void initialize() override;
    void shutdown() override;
    void calculate() override;
};

#endif
