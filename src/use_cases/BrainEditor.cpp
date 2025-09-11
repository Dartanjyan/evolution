#include "BrainEditor.h"

void BrainEditor::updateMemory(std::vector<CreaturePhysicsInputs> &data)
{
    for (auto &d : data) {
        Brain* brain = d.creature->getBrain();
        if (brain == nullptr) {
            std::cout << "BrainEditor::updateMemory: Creature has no brain\n";
            continue;
        }
        if (d.outputs.size() < brain->getMemory().size()) {
            std::cout << "BrainEditor::updateMemory: Not enough outputs to update memory (got "<<d.outputs.size()<<", expected "<<brain->getMemory().size()<<")\n";
            continue;
        }
        
        const std::vector<double> new_memory(d.outputs.end() - brain->getMemory().size(), d.outputs.end());
        const std::vector<double> new_outputs(d.outputs.begin(), d.outputs.end() - brain->getMemory().size());
        d.outputs = new_outputs;
        brain->setMemory(new_memory);
    }
}
