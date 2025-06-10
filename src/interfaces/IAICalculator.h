#ifndef IAICALCULATOR_H
#define IAICALCULATOR_H

class IAICalculator {
public:
    virtual ~IAICalculator() = default;
    
    virtual void initialize() = 0;
    virtual void shutdown() = 0;
};

#endif