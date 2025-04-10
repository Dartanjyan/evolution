#include "Creature.h"
#include <queue>
#include <algorithm>

unsigned Creature::last_id = 0;

Creature::Creature() 
    : id(Creature::newId()),
      bodyParts(std::vector<BodyPart*> {}),
      joints(std::vector<Joint*> {}),
      fitness(0.0f)
{
    std::cout << "Creating empty Creature, id = " << id << "\n";
}

Creature::Creature(std::vector<BodyPart*> bodyParts, 
                std::vector<Joint*> joints,
                Brain* brain,
                unsigned immunity): 
    id(Creature::newId()), 
    bodyParts(bodyParts),
    brain(brain),
    joints(joints),
    fitness(0.0f),
    immunity(immunity)
{
    std::cout << "Creating Creature, id = " << id << "\n";
}

Creature::Creature(const Creature &other):
    id(Creature::newId()),
    fitness(other.fitness),
    immunity(other.immunity)
{
    std::cout << "Copying Creature, id "<<other.id<<"->"<<id<< "\n";

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

    for (Joint* oldJoint : other.joints) {
        BodyPart* oldBodyA = oldJoint->getBodyA();
        BodyPart* oldBodyB = oldJoint->getBodyB();

        auto itA = partMapping.find(oldBodyA);
        auto itB = partMapping.find(oldBodyB);

        if (itA != partMapping.end() && itB != partMapping.end()) {
            BodyPart* newBodyA = itA->second;
            BodyPart* newBodyB = itB->second;

            Joint* newJoint = new Joint(
                newBodyA, newBodyB,
                oldJoint->getAnchorA(), oldJoint->getAnchorB(),
                oldJoint->getRest(), oldJoint->getStiffness(),
                oldJoint->getDamping(), oldJoint->getCollideConnected()
            );
            joints.push_back(newJoint);
        }
    }
    
    brain = new Brain(*other.brain);
}

Creature::~Creature()
{
    std::cout << "Deleting Creature, id = "<<id<<"\n";
    for (Joint* joint : joints) {
        delete joint;
    }
    joints.clear();

    for (BodyPart* part : bodyParts) {
        delete part;
    }
    joints.clear();
}

void Creature::replaceBrain(Brain *new_brain)
{
    delete brain;
    brain = new_brain;
}

void Creature::addJoint(Joint *joint)
{
    if (joint) {
        joints.push_back(joint);
    }
}

void Creature::removeJoint(Joint* joint)
{
    if (!joint) return;
    
    auto it = std::find(joints.begin(), joints.end(), joint);
    if (it != joints.end()) {
        delete *it;
        joints.erase(it);
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

unsigned Creature::newId() { return ++Creature::last_id; }

void Creature::resetId() { Creature::last_id = 0; }
