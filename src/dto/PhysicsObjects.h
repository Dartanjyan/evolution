#ifndef PHYSICS_OBJECTS_H
#define PHYSICS_OBJECTS_H

#include <vector>
#include "Vector2.h"

struct BodyObject {
    float x;
    float y;
    float angle;
    float mass;
    float inertia;
    float velocityX;
    float velocityY;
    float angularVelocity;
};

struct ShapeObject {
    float radius;
    std::vector<Vector2> vertices;
    BodyObject* body = nullptr;
};

struct ConstraintObject {
    BodyObject* partA;
    BodyObject* partB;
    float anchorAX;
    float anchorAY;
    float anchorBX;
    float anchorBY;
    float maxForce;
};

#endif