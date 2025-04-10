#ifndef BODYPART_H
#define BODYPART_H

#include <iostream>
#include <vector>
#include <algorithm>
#include "Vector2.h"

class BodyPart {
private:
    // Parent means only that new object won't have it's own new physics body
    // but only shape that'll connect to the parent's body.
    // Creature class object has vector of BodyParts
    // std::vector<BodyPart*> parts;
    BodyPart* parent = nullptr;

    unsigned id;
    unsigned previous_id = 0;
    float density;
    float friction;
    float mass;
    float elasticity;
    bool isSensorPart;
    std::vector<Vector2> vertices;
    std::vector<BodyPart*> children;

    static unsigned last_id;
    static unsigned newId();
    static void resetId();
    
public:
    BodyPart();
    BodyPart(BodyPart* parent,
            std::vector<Vector2> vertices,
            bool isSensorPart = false,
            float density = 1.0f, 
            float friction = 0.5f, 
            float mass = 1.0f, 
            float elasticity = 0.5f);
    BodyPart(const BodyPart &other, BodyPart* parent);
    ~BodyPart();

    // getters
    unsigned int getId() const { return id; }
    float getDensity() const { return density; }
    float getFriction() const { return friction; }
    float getMass() const { return mass; }
    float getElasticity() const { return elasticity; }
    bool isSensor() const { return isSensorPart; }
    std::vector<Vector2> getVertices() const { return vertices; }

    // setters
    void setDensity(float density) { this->density = density; }
    void setFriction(float friction) { this->friction = friction; }
    void setMass(float mass) { this->mass = mass; }
    void setElasticity(float elasticity) { this->elasticity = elasticity; }

    void setSensor(bool isSensor) { this->isSensorPart = isSensor; }
    void setVertices(const std::vector<Vector2>& new_vertices) { this->vertices = new_vertices; }
    void setParent(BodyPart* new_parent) { this->parent = new_parent; }

    // Parents and children stuff
    void addChild(BodyPart* child);
    void removeChild(BodyPart* child);
    BodyPart* getParent() const { return this->parent; };
    BodyPart* getRootParent() const;
    const std::vector<BodyPart*>& getChildren() const { return children; }
};

#endif
