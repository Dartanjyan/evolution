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

    parent(parent),
    id(BodyPart::newId()),
    vertices(vertices),
    radius(radius),
    density(density),
    friction(friction), 
    elasticity(elasticity),
    isSensorPart(isSensorPart)
{
    // std::cout << "Creating BodyPart, id = "<<id<< "\n";
}

BodyPart::BodyPart(const BodyPart &other, BodyPart* parent):
    parent(parent),
    id(BodyPart::newId()),
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
    switch (n) {
        case 0:
            throw std::runtime_error("BodyPart must have at least 1 vertex but has 0");
            break;
        case 1:
            area = M_PI * pow(this->radius, 2);
            break;
        case 2:
            area = M_PI * pow(this->radius, 2) + (this->radius * 2 * sqrt(pow(vertices[1].x-vertices[0].x, 2) + pow(vertices[1].y-vertices[0].y, 2)));
            break;
        default:
            for (size_t i = 0; i < n; ++i) {
                const Vector2& current = vertices[i];
                const Vector2& next = vertices[(i + 1) % n];
                area += (current.x * next.y - next.x * current.y);
            }
            area = std::abs(area) * 0.5f;
            break;
    }
    return area;
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
    float area = this->getOwnArea();
    return area * this->density;
}

float BodyPart::getMass() const
{
    float mass = this->getOwnMass();
    for (const auto& child : children) {
        mass += child->getMass();
    }
    return mass;
}

Vector2 BodyPart::getCenter() const
{
    Vector2 all_vertices = Vector2(0, 0);
    size_t amount = 0;
    for (auto vertex: this->getVertices()) {
        all_vertices = all_vertices + vertex;
        amount++;
    }
    for (auto* child : this->getAllChildren()) {
        for (auto vertex: child->getVertices()) {
            all_vertices = all_vertices + vertex;
            amount++;
        }
    }
    return all_vertices / amount;
}

std::vector<Vector2> BodyPart::getBiasedVertices() const{
    std::vector<Vector2> verts = this->vertices;
    for (auto& v : verts) {
        v = v + this->body_to_shape_bias;
    }
    return verts;
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
