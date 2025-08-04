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

    // Add physics object referring to BodyPart
    virtual void addBodyPart(unsigned creature_id, BodyPart* bodyPart) = 0;
    // Add physics constraint referring to Constraint
    virtual void addConstraint(unsigned creature_id, Constraint* constraint) = 0;
    // Add BodyParts and Constraints to build physics model of Creature
    virtual void addCreature(Creature* creature) = 0;
    // Remove physics object that refers to BodyPart* bodyPart
    virtual void removeBodyPart(unsigned creature_id, BodyPart* bodyPart) = 0;
    // Remove physics constraint that refers to Constraint* constraint
    virtual void removeConstraint(unsigned creature_id, Constraint* constraint) = 0;
    // Remove Creature from physics engine
    // This will remove all BodyParts and Constraints that refer to this Creature
    virtual void removeCreature(unsigned creature_id) = 0;

    // Get all physics objects that are used for rendering
    virtual void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) = 0;

    // Collect inputs for every creature's ai + their memory
    virtual void getCreatureAIInputs(std::vector<CreaturePhysicsInputs>& out) = 0;
    // Apply AI decisions to Creature's physics model (muscles) + save their memory
    virtual void applyAIResults(const std::vector<CreaturePhysicsInputs>& data) = 0;
};

#endif
