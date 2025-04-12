#ifndef IPHYSICSENGINE_H
#define IPHYSICSENGINE_H

#include "BodyPart.h"
#include "Joint.h"
#include "Creature.h"
#include <vector>

class IPhysicsEngine {
public:
    virtual ~IPhysicsEngine() = default;
   /*
    * initialize - create space, set up gravity etc.
    * update - do N steps
    * cleanup - delete all its objects
    *
    *
    */ 
    // Space management
    virtual void Initialize() = 0;
    virtual void Update(float deltaTime) = 0;
    virtual void Cleanup() = 0;
    
    // Object creation
    virtual void* CreatePhysicsBody(const BodyPart* bodyPart) = 0;
    virtual void* CreatePhysicsShape(const BodyPart* bodyPart, void* body) = 0;
    virtual void* CreatePhysicsJoint(const Joint* joint) = 0;
    
    // Creature management
    virtual void AddCreature(const Creature* creature) = 0;
    virtual void RemoveCreature(const Creature* creature) = 0;
    
    // Getters for position/rotation data
    virtual Vector2 GetBodyPosition(void* body) const = 0;
    virtual float GetBodyRotation(void* body) const = 0;
    
    // Physics properties
    virtual void SetGravity(Vector2 gravity) = 0;
    virtual Vector2 GetGravity() const = 0;
};

#endif
