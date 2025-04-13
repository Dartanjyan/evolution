#ifndef IPHYSICS_ENGINE_H
#define IPHYSICS_ENGINE_H

#include "BodyPart.h"
#include "Constraint.h"
#include "Vector2.h"
#include "Creature.h"
#include "PhysicsObjects.h"

class IPhysicsEngine {
public:
    virtual ~IPhysicsEngine() = default;

    // Initialize the space.
    virtual void initialize() = 0;
    virtual void update(float dt) = 0;
    virtual void shutdown() = 0;

    virtual void addBodyPart(BodyPart* bodyPart) = 0;
    virtual void addConstraint(Constraint* constraint) = 0;
    virtual void addCreature(Creature* creature) = 0;
    virtual void removeBodyPart(BodyPart* bodyPart) = 0;
    virtual void removeConstraint(Constraint* constraint) = 0;
    virtual void removeCreature(Creature* creature) = 0;

    virtual void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) const = 0;

};

#endif
