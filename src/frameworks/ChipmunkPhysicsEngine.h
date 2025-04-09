// src/frameworks/ChipmunkPhysicsEngine.h
#ifndef CHIPMUNKPHYSICSENGINE_H
#define CHIPMUNKPHYSICSENGINE_H

#include "IPhysicsEngine.h"
#include <unordered_map>
#include <map>

// Forward declarations to avoid header conflicts
struct cpSpace;
struct cpBody;
struct cpShape;
struct cpConstraint;

class ChipmunkPhysicsEngine : public IPhysicsEngine {
private:
    cpSpace* space;
    float timeStep;
    
    // Maps to keep track of physics objects
    std::unordered_map<unsigned, cpBody*> bodyPartToBody;
    std::unordered_map<unsigned, std::vector<cpShape*>> bodyPartToShapes;
    std::unordered_map<unsigned, cpConstraint*> jointToConstraint;
    std::unordered_map<unsigned, std::vector<void*>> creatureToObjects;

    // Helper methods
    void ProcessBodyPartHierarchy(const BodyPart* part, cpBody* parentBody);

public:
    ChipmunkPhysicsEngine(float timeStep = 1.0f/60.0f);
    ~ChipmunkPhysicsEngine();
    
    // IPhysicsEngine implementation
    void Initialize() override;
    void Update(float deltaTime) override;
    void Cleanup() override;
    
    void* CreatePhysicsBody(const BodyPart* bodyPart) override;
    void* CreatePhysicsShape(const BodyPart* bodyPart, void* body) override;
    void* CreatePhysicsJoint(const Joint* joint) override;
    
    void AddCreature(const Creature* creature) override;
    void RemoveCreature(const Creature* creature) override;
    
    Vector2 GetBodyPosition(void* body) const override;
    float GetBodyRotation(void* body) const override;
    
    void SetGravity(Vector2 gravity) override;
    Vector2 GetGravity() const override;

    // Get the native Chipmunk space
    cpSpace* GetNativeSpace() const;
};

#endif