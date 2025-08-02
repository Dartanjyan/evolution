#ifndef IPHYSICS_ENGINE_H
#define IPHYSICS_ENGINE_H

#include "BodyPart.h"
#include "Constraint.h"
#include "Vector2.h"
#include "Creature.h"
#include "PhysicsObjects.h"
#include "CreaturePhysicsInputs.h"

class IPhysicsEngine {
public:
    virtual ~IPhysicsEngine() = default;

    // Initialize the space.
    virtual void initialize() = 0;
    virtual void update(float dt) = 0;
    virtual void shutdown() = 0;

    virtual void addBodyPart(unsigned creature_id, BodyPart* bodyPart) = 0;
    virtual void addConstraint(unsigned creature_id, Constraint* constraint) = 0;
    virtual void addCreature(Creature* creature) = 0;
    virtual void removeBodyPart(unsigned creature_id, BodyPart* bodyPart) = 0;
    virtual void removeConstraint(unsigned creature_id, Constraint* constraint) = 0;
    virtual void removeCreature(unsigned creature_id) = 0;

    virtual void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) = 0;
    virtual void getPhysicsData(std::vector<CreaturePhysicsInputs>& out) = 0;
    virtual void applyAIResults(const std::vector<CreaturePhysicsInputs>& data) = 0;
};

#endif
