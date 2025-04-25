#ifndef BODYPART_H
#define BODYPART_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
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
    std::vector<Vector2> vertices;
    Vector2 body_to_shape_bias = Vector2();
    float radius = 0.0f;
    std::vector<BodyPart*> children;
    float density;
    float friction;
    float elasticity;
    bool isSensorPart;

    static unsigned last_id;
    static unsigned newId();
    static void resetId();
    float getOwnArea() const;
    float getOwnMass() const;
    
public:
    BodyPart();
    BodyPart(BodyPart* parent,
            std::vector<Vector2> vertices,
            bool isSensorPart = false,
            float radius = 0.0,
            float density = 0.1, 
            float friction = 1,
            float elasticity = 0.5);
    BodyPart(const BodyPart &other, BodyPart* parent);
    ~BodyPart();

    // getters
    unsigned int getId() const { return id; }
    float getDensity() const { return density; }
    float getFriction() const { return friction; }
    float getElasticity() const { return elasticity; }
    bool isSensor() const { return isSensorPart; }
    std::vector<Vector2> getVertices() const { return vertices; }
    std::vector<Vector2> getBiasedVertices() const;
    float getRadius() const { return radius; }
    float getArea() const;
    float getMass() const;
    Vector2 getCenter() const;
    Vector2 getBodyPosBias() const { return body_to_shape_bias; }

    // setters
    void setDensity(float density) { this->density = density; }
    void setFriction(float friction) { this->friction = friction; }
    void setElasticity(float elasticity) { this->elasticity = elasticity; }

    void setSensor(bool isSensor) { this->isSensorPart = isSensor; }
    void setVertices(const std::vector<Vector2>& new_vertices) { this->vertices = new_vertices; }
    void setRadius(float radius) { this->radius = radius; }
    void setParent(BodyPart* new_parent) { this->parent = new_parent; }
    void setBodyPosBias(Vector2 new_bias) { this->body_to_shape_bias = new_bias; }

    // Parents and children stuff
    void addChild(BodyPart* child);
    void removeChild(BodyPart* child);
    BodyPart* getParent() const { return this->parent; };
    BodyPart* getRootParent() const;
    const std::vector<BodyPart*>& getChildren() const { return children; }
    const std::vector<BodyPart*> getAllChildren() const;
};

#endif
