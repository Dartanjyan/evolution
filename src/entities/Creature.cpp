#include "Creature.h"
#include "print.h"
#include <queue>
#include <algorithm>

unsigned Creature::last_id = 0;

Creature::Creature() 
    : id(Creature::newId()),
      rootPart(nullptr), 
      bodyParts(std::vector<BodyPart*> {}),
      joints(std::vector<Joint*> {}),
      fitness(0.0f)
{
    print("Creature created");
}

Creature::Creature(BodyPart* rootPart, 
                std::vector<BodyPart*> bodyParts, 
                std::vector<Joint*> joints, 
                unsigned immunity): 
    id(Creature::newId()), 
    rootPart(rootPart),
    bodyParts(bodyParts),
    joints(joints),
    fitness(0.0f),
    immunity(immunity)
{
    rootPart->setRoot(true);
}

Creature::Creature(const Creature &other): 
    id(Creature::newId()),
    fitness(other.fitness),
    bodyParts(other.bodyParts),
    joints(other.joints),
    immunity(other.immunity)
{
    for (BodyPart* bodyPart: other.bodyParts) {
        BodyPart* new_part = new BodyPart(*bodyPart);
        this->bodyParts.push_back(new_part);

        if (new_part->isRoot()) { 
            this->setRootPart(new_part);
        }
    }

    for (Joint* joint : other.joints) {
        // TODO 
        /*
        При копировании Сустава копируется и ссылка на СТАРОЕ тело.
        Если ссылка на телоА у сустава равна первому телу у Старого существа, то установить ссылку на первое тело this существа.
        */
        Joint* new_joint = new Joint(*joint);
        this->joints.push_back(new_joint);
        
        auto itA = std::find(other.bodyParts.begin(), other.bodyParts.end(), joint->getBodyA());
        if (itA != other.bodyParts.end()) {
            size_t idxA = std::distance(other.bodyParts.begin(), itA);
            BodyPart* ptrBodyA = bodyParts[idxA];
            joint->setBodyA(ptrBodyA);
        }

        auto itB = std::find(other.bodyParts.begin(), other.bodyParts.end(), joint->getBodyB());
        if (itB != other.bodyParts.end()) {
            size_t idxB = std::distance(other.bodyParts.begin(), itB);
            BodyPart* ptrBodyB = bodyParts[idxB];
            joint->setBodyA(ptrBodyB);
        }

    }
}

Creature::~Creature()
{
    // TODO
    // Освобождаем память соединений
    for (Joint* joint : joints) {
        delete joint;
    }
    joints.clear();
    
    // Удаляем корневую часть (она удалит все свои дочерние части)
    if (rootPart) {
        delete rootPart;
        rootPart = nullptr;
    }
    
    print("Creature destroyed");
}

void Creature::setRootPart(BodyPart* part)
{
    if (rootPart) {
        rootPart->setRoot(false);
    }

    rootPart = part;
    if (part) {
        part->setRoot(true);
    }
}

void Creature::addJoint(Joint* joint)
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
        joints.erase(it);
        delete *it;
    }
}

/*
std::vector<BodyPart*> Creature::getAllBodyParts() const
{
    // TODO
    std::vector<BodyPart*> allParts;
    
    // Используем обход в ширину для сбора всех частей
    std::queue<BodyPart*> queue;
    queue.push(rootPart);
    
    while (!queue.empty()) {
        BodyPart* current = queue.front();
        queue.pop();
        
        allParts.push_back(current);
        
        // Добавляем всех детей в очередь
        for (BodyPart* child : current->getChildren()) {
            queue.push(child);
        }
    }
    
    return allParts;
}
*/

/*
Creature* Creature::createBasicCreature()
{
    // TODO
    // Создаем простейшее существо - тело и 4 конечности
    BodyPart* body = new BodyPart();
    body->setRoot(true);
    
    // Создаем 4 конечности (примитивные)
    BodyPart* limb1 = new BodyPart(body, std::vector<Vector2>{});
    BodyPart* limb2 = new BodyPart(body, std::vector<Vector2>{});
    BodyPart* limb3 = new BodyPart(body, std::vector<Vector2>{});
    BodyPart* limb4 = new BodyPart(body, std::vector<Vector2>{});
    
    // Добавляем конечности как дочерние части к телу
    body->addChild(limb1);
    body->addChild(limb2);
    body->addChild(limb3);
    body->addChild(limb4);
    
    // Создаем существо
    Creature* creature = new Creature(body);
    
    // Создаем соединения между частями
    Joint* joint1 = new Joint(body, limb1);
    Joint* joint2 = new Joint(body, limb2);
    Joint* joint3 = new Joint(body, limb3);
    Joint* joint4 = new Joint(body, limb4);
    
    // Добавляем соединения к существу
    creature->addJoint(joint1);
    creature->addJoint(joint2);
    creature->addJoint(joint3);
    creature->addJoint(joint4);
    
    return creature;
}
*/
unsigned Creature::newId() { return Creature::last_id++; }

void Creature::resetId() { Creature::last_id = 0; }
