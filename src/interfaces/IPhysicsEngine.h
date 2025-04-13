#ifndef IPHYSICS_ENGINE_H
#define IPHYSICS_ENGINE_H

#include "BodyPart.h"
#include "Joint.h"
#include "Vector2.h"


class IPhysicsEngine {
public:
    virtual ~IPhysicsEngine() = default;

    // Initialize the space.
    virtual void initialize() = 0;
    virtual void update(float dt) = 0;
    virtual void shutdown() = 0;

    // Add a body part to the space.
    virtual void addBodyPart(BodyPart* bodyPart) = 0;
    // Add a joint to the space.
    virtual void addJoint(Joint* joint) = 0;
    // Remove a body part from the space.
    virtual void removeBodyPart(BodyPart* bodyPart) = 0;
    // Remove a joint from the space.
    virtual void removeJoint(Joint* joint) = 0;
};

#endif
