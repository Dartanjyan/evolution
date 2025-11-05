#ifndef ISIMULATION_SAVER
#define ISIMULATION_SAVER

#include "SimulationSave.h"

class ISimulationSaver {
public:
    virtual void saveSimulation(SimulationSave save) = 0;
    SimulationSave loadSimulation() = 0;
};

#endif
