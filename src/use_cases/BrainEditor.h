#ifndef BRAINEDITOR_H
#define BRAINEDITOR_H

#include "CreaturePhysicsInputs.h"

namespace BrainEditor {
    // Save new memory to every creature's brain
    void updateMemory(std::vector<CreaturePhysicsInputs> &data);
};

#endif // BRAINEDITOR_H