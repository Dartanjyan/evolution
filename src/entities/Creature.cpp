#include "Creature.h"
#include <queue>
#include <algorithm>

unsigned Creature::last_id = 0;

Creature::Creature() 
    : id(Creature::newId()),
      bodyParts(std::vector<BodyPart*> {}),
      joints(std::vector<Constraint*> {}),
      fitness(0.0f)
{
    // std::cout << "Creating empty Creature, id = " << id << "\n";
}

Creature::Creature(std::vector<BodyPart*> bodyParts, 
                std::vector<Constraint*> joints,
                Brain* brain,
                unsigned immunity): 
    id(Creature::newId()), 
    bodyParts(bodyParts),
    brain(brain),
    joints(joints),
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

    for (Constraint* oldJoint : other.joints) {
        BodyPart* oldBodyA = oldJoint->getBodyA();
        BodyPart* oldBodyB = oldJoint->getBodyB();

        auto itA = partMapping.find(oldBodyA);
        auto itB = partMapping.find(oldBodyB);

        if (itA != partMapping.end() && itB != partMapping.end()) {
            BodyPart* newBodyA = itA->second;
            BodyPart* newBodyB = itB->second;

            Constraint* newJoint = new Constraint(
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
    // std::cout << "Deleting Creature, id = "<<id<<"\n";
    for (Constraint* joint : joints) {
        delete joint;
    }
    joints.clear();

    for (BodyPart* part : bodyParts) {
        delete part;
    }
    joints.clear();

    delete brain;
}

void Creature::replaceBrain(Brain *new_brain)
{
    delete brain;
    brain = new_brain;
}

void Creature::addConstraint(Constraint *joint)
{
    if (joint) {
        joints.push_back(joint);
    }
}

void Creature::removeConstraint(Constraint* joint)
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

Creature* Creature::createBasicCreature()
{
    std::vector<Vector2> bodyVertices = {
        Vector2(-1.0f, -1.0f),
        Vector2( 1.0f, -1.0f),
        Vector2( 1.0f,  1.0f),
        Vector2(-1.0f,  1.0f)
    };
    BodyPart* body = new BodyPart(nullptr, bodyVertices);
    
    std::vector<Vector2> limbVertices = {
        Vector2(0.0f, 0.0f),
        Vector2(0.5f, -1.0f),
        Vector2(-0.5f, -1.0f)
    };
    
    BodyPart* limb1 = new BodyPart(nullptr, limbVertices);
    BodyPart* limb2 = new BodyPart(nullptr, limbVertices);
    BodyPart* limb3 = new BodyPart(nullptr, limbVertices);
    BodyPart* limb4 = new BodyPart(nullptr, limbVertices);
    
    std::vector<BodyPart*> bodyParts = {body, limb1, limb2, limb3, limb4};
    
    Constraint* joint1 = new Constraint(body, limb1, Vector2(0.0f, -1.0f), Vector2(0.0f, 0.0f));
    Constraint* joint2 = new Constraint(body, limb2, Vector2(1.0f, 0.0f), Vector2(0.0f, 0.0f));
    Constraint* joint3 = new Constraint(body, limb3, Vector2(0.0f, 1.0f), Vector2(0.0f, 0.0f));
    Constraint* joint4 = new Constraint(body, limb4, Vector2(-1.0f, 0.0f), Vector2(0.0f, 0.0f));
    
    std::vector<Constraint*> joints = {joint1, joint2, joint3, joint4};
    
    Brain* brain = new Brain(std::vector<unsigned short>{1}, std::vector<std::vector<double>>{
        {0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5, 0.5}
    }, std::vector<double>{0.5, 0.5, 0.5, 0.5});

    Creature* creature = new Creature(bodyParts, joints, brain);
    
    return creature;
}

unsigned Creature::newId() { return ++Creature::last_id; }

void Creature::resetId() { Creature::last_id = 0; }
