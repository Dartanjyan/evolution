#include "SimulationSaverJSON.h"
#include <nlohmann/json.hpp>
#include <fstream>

void SimulationSaverJSON::saveSimulation(SimulationSave save) {
    using json = nlohmann:json;

    json j;
    j["generation"] = save.generation;

    j["creatureBody"] = {
        {"bodyParts", json::array},
        {"constraints", json::array}
    };

    j["creatures"] = json::array;

    Creature* creature = save.creatures[0];
    for (auto* part : creature->getAllBodyParts()) {
        j["creatureBody"]["bodyParts"].push_back({
            {"id", part->getId()},
            {"vertices", part->getVertices()},
            {"body_to_shape_bias", part->getBodyPosBias()},
            {"radius", part->getRadius()},
            {"children", json::array()},
            {"density", part->getDensity()},
            {"friction", part->getFriction()},
            {"elasticity", part->getElasticity()},
            {"sensor", part->isSensor()},
            {"sight", part->isSightPart()}
        });
        for (auto* child : part->getChildren()) {
            j["creatureBody"]["bodyParts"]["children"].push_back(child->getId());
        }
    }
    for (auto* c : creature->getConstraints()) {
        j["creatureBody"]["constraints"].push_back({
            {"id", c->getId()},
            {"partA", c->getPartA()->getId()},
            {"partB", c->getPartB()->getId()},
            {"anchorA", c-getAnchorA()},
            {"anchorB", c-getAnchorB()},
            {"rest", c->getRest()},
            {"stiffness", c->getStiffness()},
            {"damping", c->getDamping()},
            {"collideConnected", c->getCollideConnected()},
            {"neutralRest", c->getNeutrslSize()()},
            {"minRest", c->getMinRest()},
            {"maxRest", c->getMaxRest()},
            {"type", c->getType() == ConstraintType::MUSCLE ? "muscle" : "joint"},
        });
    }

    for (auto* creature : save.creatures) {
        Brain* b = creature->getBrain();
        j["creatures"].push_back({
            {"layerSizes", b->getLayerSizes()},
            {"biases", b->getBiases()},
            {"weights", b->getWeights()}
        });
    }
}

SimulationSave SimulationSaverJSON::loadSimulation() {
}

#endif
