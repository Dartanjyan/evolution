#ifndef PHYSICS_OBJECTS_H
#define PHYSICS_OBJECTS_H

#include <vector>
#include "Vector2.h"
#include "Constraint.h"

class BodyObject {
public:
    BodyObject() = default;
    BodyObject(unsigned id, Vector2 position, float angle, float mass, Vector2 velocity);
    Vector2 position {Vector2()};
    float angle {0};
    float mass {0};
    // float inertia;
    Vector2 velocity {Vector2()};
    // float angularVelocity;
    unsigned id {0};
};

class ShapeObject {
public:
    ShapeObject() = default;
    ShapeObject(unsigned id, float radius, std::vector<Vector2> vertices, BodyObject* body);
    float radius {0.0f};
    std::vector<Vector2> vertices {};
    BodyObject* body {nullptr};
    unsigned id {0};
};

class ConstraintObject {
public:
    BodyObject* partA {nullptr};
    BodyObject* partB {nullptr};
    Vector2 anchorA {Vector2()};
    Vector2 anchorB {Vector2()};
    ConstraintType constraintType;
    unsigned id {0};
    // float maxForce;
};

#endif
