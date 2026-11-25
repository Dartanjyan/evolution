#ifndef SIMULATION_SAVER_JSON
#define SIMULATION_SAVER_JSON

#include "ISimulationSaver.h"

class SimulationSaverJSON : public ISimulationSaver {
public:
    SimulationSaverJSON();

    void saveSimulation(SimulationSave save) override;
    LoadedSimulation loadSimulation() override;
};

#endif
