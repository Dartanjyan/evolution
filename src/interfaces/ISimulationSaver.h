#ifndef ISIMULATION_SAVER
#define ISIMULATION_SAVER

#include "SimulationSave.h"

class ISimulationSaver {
public:
    virtual ~ISimulationSaver() = default;
    
    virtual void saveSimulation(SimulationSave save) = 0;
    virtual SimulationSave loadSimulation() = 0;
};

#endif
