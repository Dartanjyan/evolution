#include "Creature.h"
#include <queue>
#include <algorithm>

unsigned Creature::last_id = 0;

Creature::Creature() 
    : id(Creature::newId()),
      bodyParts(std::vector<BodyPart*> {}),
      constraints(std::vector<Constraint*> {}),
      fitness(0.0f)
{
    // std::cout << "Creating empty Creature, id = " << id << "\n";
}

Creature::Creature(std::vector<BodyPart*> bodyParts, 
                std::vector<Constraint*> constraints,
                Brain* brain,
                unsigned immunity): 
    id(Creature::newId()), 
    bodyParts(bodyParts),
    brain(brain),
    constraints(constraints),
    fitness(0.0f),
    immunity(immunity)
{
    // std::cout << "Creating Creature, id = " << id << "\n";
}

Creature::Creature(const Creature &other):
    id(Creature::newId()),
    fitness(other.fitness),
    immunity(other.immunity)
{
    // std::cout << "Copying Creature, id "<<other.id<<"->"<<id<< "\n";

    std::map<BodyPart*, BodyPart*> partMapping;

    for (BodyPart* oldPart : other.bodyParts) {
        BodyPart* newPart = new BodyPart(*oldPart, nullptr);
        bodyParts.push_back(newPart);

        partMapping[oldPart] = newPart;

        std::function<void(BodyPart*, BodyPart*)> mapBodyPartsRecursively =
            [&partMapping, &mapBodyPartsRecursively](BodyPart* oldPart, BodyPart* newPart) {
                partMapping[oldPart] = newPart;

                if (oldPart->getChildren().size() != newPart->getChildren().size()) {
                    return;
                }

                for (size_t i = 0; i < oldPart->getChildren().size(); ++i) {
                    mapBodyPartsRecursively(oldPart->getChildren()[i], newPart->getChildren()[i]);
                }
            };

        mapBodyPartsRecursively(oldPart, newPart);
    }

    for (Constraint* oldConstraint : other.constraints) {
        BodyPart* oldBodyA = oldConstraint->getPartA();
        BodyPart* oldBodyB = oldConstraint->getPartB();

        auto itA = partMapping.find(oldBodyA);
        auto itB = partMapping.find(oldBodyB);

        if (itA != partMapping.end() && itB != partMapping.end()) {
            BodyPart* newBodyA = itA->second;
            BodyPart* newBodyB = itB->second;

            Constraint* newConstraint = new Constraint(
                newBodyA, newBodyB, oldConstraint->getType(),
                oldConstraint->getAnchorA(), oldConstraint->getAnchorB(),
                oldConstraint->getRest(), oldConstraint->getStiffness(),
                oldConstraint->getDamping(), oldConstraint->getCollideConnected()
            );
            constraints.push_back(newConstraint);
        }
    }
    
    brain = new Brain(*other.brain);
}

Creature::~Creature()
{
    // std::cout << "Deleting Creature, id = "<<id<<"\n";
    for (Constraint* constraint : constraints) {
        delete constraint;
    }
    constraints.clear();

    for (BodyPart* part : bodyParts) {
        delete part;
    }
    constraints.clear();

    delete brain;
}

void Creature::replaceBrain(Brain *new_brain)
{
    delete brain;
    brain = new_brain;
}

void Creature::addConstraint(Constraint *constraint)
{
    if (constraint) {
        constraints.push_back(constraint);
    }
}

void Creature::removeConstraint(Constraint* constraint)
{
    if (!constraint) return;
    
    auto it = std::find(constraints.begin(), constraints.end(), constraint);
    if (it != constraints.end()) {
        delete *it;
        constraints.erase(it);
    }
}

std::vector<BodyPart*> Creature::getAllBodyParts() const
{
    std::vector<BodyPart*> allParts;
    
    std::function<void(BodyPart*)> collectParts = [&allParts, &collectParts](BodyPart* part) {
        allParts.push_back(part);
        
        for (BodyPart* child : part->getChildren()) {
            collectParts(child);
        }
    };
    
    for (BodyPart* rootPart : bodyParts) {
        collectParts(rootPart);
    }
    
    return std::vector<BodyPart*>(allParts);
}

const BodyPart* Creature::getBodyPartById(unsigned id) const
{
    for (auto* bodyPart : this->getAllBodyParts()) {
        if (bodyPart->getId() == id) { 
            return bodyPart;
        }
    }
    return nullptr;
}

Creature* Creature::createBasicCreature()
{
    float scale = 10;
    Vector2 bias(100, 100);
    std::vector<Vector2> bodyVertices = {
        Vector2(-1.0f, -1.0f)*scale+bias,
        Vector2( 1.0f, -1.0f)*scale+bias,
        Vector2( 1.0f,  1.0f)*scale+bias,
        Vector2(-1.0f,  1.0f)*scale+bias
    };
    BodyPart* body = new BodyPart(nullptr, bodyVertices);
    
    std::vector<Vector2> limbVertices = {
        Vector2(0.0f, 0.0f)*scale+bias,
        Vector2(0.5f, -1.0f)*scale+bias,
        Vector2(-0.5f, -1.0f)*scale+bias
    };
    
    BodyPart* limb1 = new BodyPart(nullptr, limbVertices);
    BodyPart* limb2 = new BodyPart(nullptr, limbVertices);
    BodyPart* limb3 = new BodyPart(nullptr, limbVertices);
    BodyPart* limb4 = new BodyPart(nullptr, limbVertices);
    
    std::vector<BodyPart*> bodyParts = {body, limb1, limb2, limb3, limb4};
    
    Constraint* constraint1 = new Constraint(body, limb1, ConstraintType::JOINT, Vector2(0.0f, -1.0f)*scale+bias, Vector2(0.0f, 0.0f)*scale+bias);
    Constraint* constraint2 = new Constraint(body, limb2, ConstraintType::JOINT, Vector2(1.0f, 0.0f)*scale+bias, Vector2(0.0f, 0.0f)*scale+bias);
    Constraint* constraint3 = new Constraint(body, limb3, ConstraintType::JOINT, Vector2(0.0f, 1.0f)*scale+bias, Vector2(0.0f, 0.0f)*scale+bias);
    Constraint* constraint4 = new Constraint(body, limb4, ConstraintType::JOINT, Vector2(-1.0f, 0.0f)*scale+bias, Vector2(0.0f, 0.0f)*scale+bias);
    
    std::vector<Constraint*> constraints = {constraint1, constraint2, constraint3, constraint4};
    
    Brain* brain = new Brain(std::vector<unsigned short>{1}, std::vector<std::vector<double>>{
        {0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5}
    }, std::vector<double>{0.5, 0.5, 0.5, 0.5});

    Creature* creature = new Creature(bodyParts, constraints, brain);
    
    return creature;
}

unsigned Creature::newId() { return ++Creature::last_id; }

void Creature::resetId() { Creature::last_id = 0; }
