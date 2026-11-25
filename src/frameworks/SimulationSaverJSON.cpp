#include "SimulationSaverJSON.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include "Vector2JSONConverter.h"

namespace {
using nlohmann::json;
constexpr char saveFilename[] = "simulation_save.json";

unsigned readUnsigned(const json& value)
{
    if (!value.is_number_unsigned() || value.get<json::number_unsigned_t>() > std::numeric_limits<unsigned>::max())
        throw std::runtime_error("Expected an unsigned integer");
    return value.get<unsigned>();
}

float readFloat(const json& value)
{
    const auto number = value.get<float>();
    if (!std::isfinite(number))
        throw std::runtime_error("Expected a finite number");
    return number;
}

std::unique_ptr<Creature> readBody(const json& body)
{
    const auto& parts = body.at("bodyParts").get_ref<const json::array_t&>();
    std::map<unsigned, const json*> partsById;
    std::set<unsigned> childIds;
    for (const auto& part : parts) {
        if (!partsById.emplace(readUnsigned(part.at("id")), &part).second)
            throw std::runtime_error("Duplicate body part ID");
        for (const auto& childId : part.at("children").get_ref<const json::array_t&>()) {
            if (!childIds.insert(readUnsigned(childId)).second)
                throw std::runtime_error("A body part has multiple parent references");
        }
    }

    std::map<unsigned, BodyPart*> loadedParts;
    const auto readPart = [&](const auto& readPart, unsigned id, BodyPart* parent) -> std::unique_ptr<BodyPart> {
        const auto& data = *partsById.at(id);
        auto part = std::make_unique<BodyPart>(parent, data.at("vertices").get<std::vector<Vector2>>(),
            data.at("sensor").get<bool>(), readFloat(data.at("radius")),
            readFloat(data.at("density")), readFloat(data.at("friction")),
            readFloat(data.at("elasticity")));
        part->setBodyPosBias(data.at("body_to_shape_bias").get<Vector2>());
        part->setSightPart(data.at("sight").get<bool>());
        if (!(part->getRadius() >= 0 && part->getDensity() > 0 &&
              part->getFriction() >= 0 && part->getElasticity() >= 0 &&
              part->getArea() > 0 && std::isfinite(part->getMass())))
            throw std::runtime_error("Invalid body part geometry or material");

        loadedParts.emplace(id, part.get());
        for (const auto& childId : data.at("children")) {
            auto child = readPart(readPart, readUnsigned(childId), part.get());
            part->addChild(child.get());
            child.release();
        }
        return part;
    };

    std::vector<std::unique_ptr<BodyPart>> ownedParts;
    std::vector<BodyPart*> roots;
    for (const auto& part : parts) {
        const auto id = readUnsigned(part.at("id"));
        if (!childIds.contains(id)) {
            ownedParts.push_back(readPart(readPart, id, nullptr));
            roots.push_back(ownedParts.back().get());
        }
    }
    if (roots.empty() || loadedParts.size() != parts.size())
        throw std::runtime_error("Body parts must form a nonempty tree or forest");

    std::vector<std::unique_ptr<Constraint>> ownedConstraints;
    std::vector<Constraint*> constraints;
    for (const auto& data : body.at("constraints").get_ref<const json::array_t&>()) {
        auto* partA = loadedParts.at(readUnsigned(data.at("partA")));
        auto* partB = loadedParts.at(readUnsigned(data.at("partB")));
        if (partA == partB || partA->getParent() || partB->getParent())
            throw std::runtime_error("A constraint must connect two distinct physics bodies");

        const auto type = data.at("type").get<std::string>();
        if (type != "joint" && type != "muscle")
            throw std::runtime_error("Unknown constraint type");
        auto constraint = std::make_unique<Constraint>(partA, partB,
            type == "muscle" ? ConstraintType::MUSCLE : ConstraintType::JOINT,
            data.at("anchorA").get<Vector2>(), data.at("anchorB").get<Vector2>(),
            readFloat(data.at("rest")), readFloat(data.at("stiffness")),
            readFloat(data.at("damping")), data.at("collideConnected").get<bool>());
        // Older saves can contain uninitialized values for these unused joint fields.
        if (constraint->getType() == ConstraintType::MUSCLE) {
            const auto neutral = readFloat(data.at("neutralRest"));
            const auto minimum = readFloat(data.at("minRest"));
            const auto maximum = readFloat(data.at("maxRest"));
            if (neutral < 0 || minimum > maximum)
                throw std::runtime_error("Invalid muscle rest lengths");
            constraint->setRestLimits(neutral, minimum, maximum);
        }
        constraints.push_back(constraint.get());
        ownedConstraints.push_back(std::move(constraint));
    }

    auto creature = std::make_unique<Creature>(roots, constraints, nullptr);
    for (auto& part : ownedParts)
        part.release();
    for (auto& constraint : ownedConstraints)
        constraint.release();
    return creature;
}

std::unique_ptr<Brain> readBrain(const json& data, Creature& creature)
{
    std::vector<std::size_t> layers;
    for (const auto& layer : data.at("layerSizes").get_ref<const json::array_t&>())
        layers.push_back(readUnsigned(layer));
    const auto weights = data.at("weights").get<std::vector<std::vector<double>>>();
    const auto biases = data.at("biases").get<std::vector<double>>();
    if (layers.size() < 2 || weights.size() != layers.size() - 1 || biases.size() != weights.size())
        throw std::runtime_error("Invalid brain dimensions");
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (!layers[i] || !layers[i + 1] ||
            weights[i].size() / layers[i] != layers[i + 1] || weights[i].size() % layers[i] != 0)
            throw std::runtime_error("Invalid brain weight dimensions");
    }

    const auto muscles = creature.getMuscles().size();
    if (layers.back() < muscles)
        throw std::runtime_error("Brain has too few outputs for the saved body");
    const auto memorySize = layers.back() - muscles;
    const auto inputsWithoutSight = 2 + creature.getJoints().size() * 2 + memorySize;
    const auto sightParts = creature.getSightParts();
    // Older saves could mark parts as sight sensors even for brains with no sight inputs.
    if (layers.front() == inputsWithoutSight) {
        for (auto* part : sightParts)
            part->setSightPart(false);
    } else if (layers.front() != inputsWithoutSight + sightParts.size() * 5) {
        throw std::runtime_error("Brain input size does not match the saved body");
    }
    return std::make_unique<Brain>(layers, memorySize, weights, biases);
}
}

SimulationSaverJSON::SimulationSaverJSON()
{
}

void SimulationSaverJSON::saveSimulation(SimulationSave save)
{
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

    std::cout << "Saving simulation to " << saveFilename << "... ";
    std::ofstream file(saveFilename);
    if (!file.is_open()) {
        std::cerr << "Couldn't open file to save simulation!" << std::endl;
        return;
    }
    file << j.dump(2);
    std::cout << "Done!" << std::endl;
    file.close();
}

LoadedSimulation SimulationSaverJSON::loadSimulation() {
    std::ifstream file(saveFilename);
    if (!file)
        throw std::runtime_error("Could not open the save file");

    const auto data = json::parse(file);
    LoadedSimulation save{readUnsigned(data.at("generation")), {}};
    const auto& population = data.at("creatures").get_ref<const json::array_t&>();
    if (population.empty())
        throw std::runtime_error("The save file contains no creatures");

    auto first = readBody(data.at("creatureBody"));
    first->replaceBrain(readBrain(population.front(), *first).release());
    save.creatures.reserve(population.size());
    save.creatures.push_back(std::move(first));
    for (std::size_t i = 1; i < population.size(); ++i) {
        const auto& source = save.creatures.front();
        auto brain = readBrain(population[i], *source);
        if (brain->getLayerSizes() != source->getBrain()->getLayerSizes())
            throw std::runtime_error("Saved brains have different layer sizes");
        save.creatures.push_back(source->cloneWithBrain(std::move(brain)));
    }
    return save;
}
