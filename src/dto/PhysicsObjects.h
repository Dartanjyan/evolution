#ifndef PHYSICS_OBJECTS_H
#define PHYSICS_OBJECTS_H

#include <vector>
#include "Vector2.h"
#include "Constraint.h"

enum class ShapeType {
    Circle, 
    Segment, 
    Polygon
};

class BodyObject {
public:
    unsigned id;
    Vector2 position;
    float angle;
    float mass;
    // float inertia;
    Vector2 velocity;
    // float angularVelocity;
    BodyObject() = default;
    BodyObject(unsigned id, Vector2 position, float angle, float mass, Vector2 velocity);
};

class ShapeObject {
public:
    unsigned id;
    float radius;
    std::vector<Vector2> vertices;
    BodyObject* body;
    ShapeType shapeType;
    bool isWorldObj = false;
    ShapeObject() = default;
    ShapeObject(unsigned id, float radius, std::vector<Vector2> vertices, BodyObject* body, bool isWorldObj);
    void initShapeType();
};

class ConstraintObject {
public:
    BodyObject* partA;
    BodyObject* partB;
    Vector2 anchorA;
    Vector2 anchorB;
    ConstraintType constraintType;
    unsigned id;
};

#endif
