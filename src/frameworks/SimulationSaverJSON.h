#ifndef SIMULATION_SAVER_JSON
#define SIMULATION_SAVER_JSON

#include "ISimulationSaver.h"

class SimulationSaverJSON : public ISimulationSaver {
public:
    void saveSimulation(SimulationSave save) override;
    SimulationSave loadSimulation() override;
};

#endif
