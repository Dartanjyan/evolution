#ifndef PHYSICS_OBJECTS_H
#define PHYSICS_OBJECTS_H

#include <vector>
#include "Vector2.h"
#include "Constraint.h"

struct BodyObject {
    Vector2 position;
    float angle;
    float mass;
    // float inertia;
    Vector2 velocity;
    // float angularVelocity;
    unsigned id;
};

struct ShapeObject {
    float radius;
    std::vector<Vector2> vertices;
    BodyObject* body;
    unsigned id;
};

struct ConstraintObject {
    BodyObject* partA;
    BodyObject* partB;
    Vector2 anchorA;
    Vector2 anchorB;
    ConstraintType constraintType;
    unsigned id;
    // float maxForce;
};

#endif
