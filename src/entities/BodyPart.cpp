#include "BodyPart.h"

unsigned BodyPart::last_id = 0;

BodyPart::BodyPart(): 
    BodyPart::BodyPart(nullptr, std::vector<Vector2>{})
{
    id = BodyPart::newId();
    std::cout << "Created empty BodyPart, id = "<<id<< "\n";
}

BodyPart::BodyPart(
    BodyPart* parent,
    std::vector<Vector2> vertices,
    bool isSensorPart,
    float density, 
    float friction, 
    float mass, 
    float elasticity):

    id(BodyPart::newId()),
    parent(parent),
    vertices(vertices),
    isSensorPart(isSensorPart),
    density(density),
    friction(friction), 
    mass(mass), 
    elasticity(elasticity)
{
    std::cout << "Creating BodyPart, id = "<<id<< "\n";
}

BodyPart::BodyPart(const BodyPart &other, BodyPart* parent):
    id(BodyPart::newId()),
    parent(parent),
    vertices(other.vertices),
    density(other.density), 
    friction(other.friction),
    mass(other.mass),
    elasticity(other.elasticity),
    isSensorPart(other.isSensorPart)
{
    std::cout << "Copying BodyPart, id "<<other.id<<"->"<<id<< "\n";

    for(auto child : other.children) {
        children.push_back(new BodyPart(*child));
    }
}

BodyPart::~BodyPart()
{
    std::cout << "Deleting BodyPart, id = "<<id<<", which has ";
    if (children.size() > 0) { 
        std::cout << children.size(); 
        if (children.size()%10 == 1) { 
            std::cout << " child"; 
        } else {
            std::cout << " children";
        }
    } else {
        std::cout << "no children";
    }
    std::cout << "\n";


    for(auto* child: children) {
        delete child;
    }
}

void BodyPart::addChild(BodyPart* child) { children.push_back(child); }
void BodyPart::removeChild(BodyPart *child) { children.erase(std::find(children.begin(), children.end(), child)); }

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

unsigned BodyPart::newId() { return ++BodyPart::last_id; }
void BodyPart::resetId() { BodyPart::last_id = 0; }
