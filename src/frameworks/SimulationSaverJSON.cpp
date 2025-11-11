#include "SimulationSaverJSON.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include "Vector2JSONConverter.h"

SimulationSaverJSON::SimulationSaverJSON()
{
}

void SimulationSaverJSON::saveSimulation(SimulationSave save)
{
    using json = nlohmann::json;

    json j;
    j = json{
        {"generation", save.generation},
        {"creatureBody", json{
            {"bodyParts", json::array()},
            {"constraints", json::array()}
        }},
        {"creatures", json::array()}
    };

    Creature* creature = save.creatures[0];
    for (auto* part : creature->getAllBodyParts()) {
        json part_json = json::object();
        part_json["id"] = part->getId();
        part_json["vertices"] = part->getVertices();
        part_json["body_to_shape_bias"] = part->getBodyPosBias();
        part_json["radius"] = part->getRadius();
        part_json["children"] = json::array();
        part_json["density"] = part->getDensity();
        part_json["friction"] = part->getFriction();
        part_json["elasticity"] = part->getElasticity();
        part_json["sensor"] = part->isSensor();
        part_json["sight"] = part->isSightPart();
        for (auto* child : part->getChildren()) {
            part_json["children"].push_back(child->getId());
        }

        j["creatureBody"]["bodyParts"].push_back(part_json);
    }

    for (auto* c : creature->getConstraints()) {
        j["creatureBody"]["constraints"].push_back(json{
            {"id", c->getId()},
            {"partA", c->getPartA()->getId()},
            {"partB", c->getPartB()->getId()},
            {"anchorA", c->getAnchorA()},
            {"anchorB", c->getAnchorB()},
            {"rest", c->getRest()},
            {"stiffness", c->getStiffness()},
            {"damping", c->getDamping()},
            {"collideConnected", c->getCollideConnected()},
            {"neutralRest", c->getNeutralSize()},
            {"minRest", c->getMinRest()},
            {"maxRest", c->getMaxRest()},
            {"type", c->getType() == ConstraintType::MUSCLE ? "muscle" : "joint"}
        });
    }

    for (auto* creature : save.creatures) {
        Brain* b = creature->getBrain();
        j["creatures"].push_back(json{
            {"layerSizes", b->getLayerSizes()},
            {"biases", b->getBiases()},
            {"weights", b->getWeights()}
        });
    }

    std::ofstream file("simulation_save.json");
    file << j.dump(2);
}

SimulationSave SimulationSaverJSON::loadSimulation() {
    // construct SimulationSave with explicit values to avoid using deleted default constructor
    SimulationSave save{0, std::vector<Creature*>()};
    return save;
}
