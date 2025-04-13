#include "BodyPart.h"

unsigned BodyPart::last_id = 0;

BodyPart::BodyPart(): 
    BodyPart::BodyPart(nullptr, std::vector<Vector2>{})
{
    id = BodyPart::newId();
    // std::cout << "Created empty BodyPart, id = "<<id<< "\n";
}

BodyPart::BodyPart(
    BodyPart* parent,
    std::vector<Vector2> vertices,
    bool isSensorPart,
    float radius,
    float density, 
    float friction, 
    float elasticity):

    id(BodyPart::newId()),
    parent(parent),
    vertices(vertices),
    radius(radius),
    isSensorPart(isSensorPart),
    density(density),
    friction(friction), 
    elasticity(elasticity)
{
    // std::cout << "Creating BodyPart, id = "<<id<< "\n";
}

BodyPart::BodyPart(const BodyPart &other, BodyPart* parent):
    id(BodyPart::newId()),
    parent(parent),
    vertices(other.vertices),
    radius(other.radius),
    density(other.density), 
    friction(other.friction),
    elasticity(other.elasticity),
    isSensorPart(other.isSensorPart)
{
    // std::cout << "Copying BodyPart, id "<<other.id<<"->"<<id<< "\n";

    for(auto child : other.children) {
        children.push_back(new BodyPart(*child));
    }
}

BodyPart::~BodyPart()
{
    /*
    // std::cout << "Deleting BodyPart, id = "<<id<<", which has ";
    if (children.size() > 0) { 
        // std::cout << children.size(); 
        if (children.size()%10 == 1) { 
            // std::cout << " child"; 
        } else {
            // std::cout << " children";
        }
    } else {
        // std::cout << "no children";
    }
    // std::cout << "\n";
    */

    for(auto* child: children) {
        delete child;
    }
}

float BodyPart::getOwnArea() const {
    float area = 0.0f;
    size_t n = vertices.size();
    for (size_t i = 0; i < n; ++i) {
        const Vector2& current = vertices[i];
        const Vector2& next = vertices[(i + 1) % n];
        area += (current.x * next.y - next.x * current.y);
    }
    return std::abs(area) * 0.5f;
}

float BodyPart::getArea() const
{
    float area = this->getOwnArea();
    for (const auto& child : children) {
        area += child->getArea();
    }
    return area;
}

float BodyPart::getOwnMass() const
{
    return this->getOwnArea() * this->density;
}

float BodyPart::getMass() const
{
    float mass = this->getOwnMass();
    for (const auto& child : children) {
        mass += child->getMass();
    }
    return mass;
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

const std::vector<BodyPart *> BodyPart::getAllChildren() const
{
    std::vector<BodyPart *> all_children = children;
    for (const auto& child : children) {
        const auto& child_children = child->getAllChildren();
        all_children.insert(all_children.end(), child_children.begin(), child_children.end());
    }
    return all_children;
}

unsigned BodyPart::newId() { return ++BodyPart::last_id; }
void BodyPart::resetId() { BodyPart::last_id = 0; }
