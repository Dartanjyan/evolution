#include "BodyPart.h"

unsigned BodyPart::last_id = 0;

BodyPart::BodyPart(): 
    BodyPart::BodyPart(nullptr, std::vector<Vector2>{})
{
    id = BodyPart::newId();
}

BodyPart::BodyPart(
    BodyPart* parent,
    std::vector<Vector2> vertices,
    bool isRootPart,
    bool isSensorPart,
    float density, 
    float friction, 
    float mass, 
    float elasticity):

    id(BodyPart::newId()),
    parent(parent),
    vertices(vertices),
    isRootPart(isRootPart), 
    isSensorPart(isSensorPart),
    density(density),
    friction(friction), 
    mass(mass), 
    elasticity(elasticity) {}

BodyPart::BodyPart(const BodyPart &other):
    id(BodyPart::newId()),
    parent(other.parent),
    vertices(other.vertices),
    density(other.density), 
    friction(other.friction),
    mass(other.mass),
    elasticity(other.elasticity),
    isRootPart(other.isRootPart), 
    isSensorPart(other.isSensorPart) {}

BodyPart::~BodyPart()
{
    for(auto* child: children) {
        delete child;
    }
}

void BodyPart::addChild(BodyPart* child) { children.push_back(child); }
void BodyPart::removeChild(BodyPart *child) { children.erase(this->getChildIter(child)); }

std::vector<BodyPart*>::iterator BodyPart::getChildIter(BodyPart *child)
{
    for (auto it = this->children.begin(); it != this->children.end(); ++it) {
        if (*it == child) {
            return it;
        }
    }
    // If not found, return end iterator
    // This is a bit of a hack, but we need to return an iterator
    // TODO throw exception if child not found
    return std::vector<BodyPart*>::iterator();
}

BodyPart *BodyPart::getRootParent() const
{
    if (this->parent != nullptr) { 
        return this->parent->getRootParent(); 
    } 
    else {
        // Return pointer to itself
        // Need to cast pointer because `this` is a const pointer
        return const_cast<BodyPart*>(this);
    }
}

unsigned BodyPart::newId() { return BodyPart::last_id++; }
void BodyPart::resetId() { BodyPart::last_id = 0; }
