// src/interfaces/IPhysicsEngine.h
#ifndef IPHYSICSENGINE_H
#define IPHYSICSENGINE_H

#include "../entities/BodyPart.h"
#include "../entities/Joint.h"
#include "../entities/Creature.h"
#include <vector>

class IPhysicsEngine {
public:
    virtual ~IPhysicsEngine() = default;
    
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