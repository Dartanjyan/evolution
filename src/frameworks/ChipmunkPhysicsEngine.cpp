// src/frameworks/ChipmunkPhysicsEngine.cpp
#include "ChipmunkPhysicsEngine.h"
#include <chipmunk/chipmunk.h>
#include <iostream>

ChipmunkPhysicsEngine::ChipmunkPhysicsEngine(float timeStep) : timeStep(timeStep), space(nullptr) {
    Initialize();
}

ChipmunkPhysicsEngine::~ChipmunkPhysicsEngine() {
    Cleanup();
}

void ChipmunkPhysicsEngine::Initialize() {
    if (space == nullptr) {
        space = cpSpaceNew();
        // Default gravity pointing down
        cpSpaceSetGravity(space, cpv(0, 9.8));
    }
}

void ChipmunkPhysicsEngine::Cleanup() {
    if (space) {
        // Clean up constraints
        for (auto& pair : jointToConstraint) {
            cpSpaceRemoveConstraint(space, pair.second);
            cpConstraintFree(pair.second);
        }
        jointToConstraint.clear();

        // Clean up shapes
        for (auto& pair : bodyPartToShapes) {
            for (cpShape* shape : pair.second) {
                cpSpaceRemoveShape(space, shape);
                cpShapeFree(shape);
            }
        }
        bodyPartToShapes.clear();

        // Clean up bodies
        for (auto& pair : bodyPartToBody) {
            cpSpaceRemoveBody(space, pair.second);
            cpBodyFree(pair.second);
        }
        bodyPartToBody.clear();

        // Free the space
        cpSpaceFree(space);
        space = nullptr;
    }
    
    creatureToObjects.clear();
}

void ChipmunkPhysicsEngine::Update(float deltaTime) {
    if (space) {
        // Fixed timestep for stability
        cpSpaceStep(space, timeStep);
    }
}

void* ChipmunkPhysicsEngine::CreatePhysicsBody(const BodyPart* bodyPart) {
    if (!bodyPart || !space) return nullptr;
    
    // Calculate basic physical properties
    float mass = bodyPart->getMass();
    float moment = cpMomentForBox(mass, 2.0f, 2.0f); // Simple approximation
    
    // Create the body
    cpBody* body = cpBodyNew(mass, moment);
    
    // Store in our map and add to space
    bodyPartToBody[bodyPart->getId()] = body;
    cpSpaceAddBody(space, body);
    
    return body;
}

void* ChipmunkPhysicsEngine::CreatePhysicsShape(const BodyPart* bodyPart, void* body) {
    if (!bodyPart || !body || !space) return nullptr;
    
    cpBody* cp_body = static_cast<cpBody*>(body);
    cpShape* shape = nullptr;
    
    // Get the vertices
    std::vector<Vector2> vertices = bodyPart->getVertices();
    
    if (vertices.size() >= 3) {
        // Create a temporary array of cpVect for the polygon
        cpVect* verts = new cpVect[vertices.size()];
        for (size_t i = 0; i < vertices.size(); i++) {
            verts[i] = cpv(vertices[i].x, vertices[i].y);
        }
        
        // Create the polygon shape
        shape = cpPolyShapeNew(cp_body, vertices.size(), verts, cpTransformIdentity, 0.0);
        
        delete[] verts;
    } else {
        // Default to a circle if not enough vertices
        shape = cpCircleShapeNew(cp_body, 1.0, cpv(0, 0));
    }
    
    // Set physics properties
    cpShapeSetFriction(shape, bodyPart->getFriction());
    cpShapeSetElasticity(shape, bodyPart->getElasticity());
    
    // Handle sensor flag
    if (bodyPart->isSensor()) {
        cpShapeSetSensor(shape, cpTrue);
    }
    
    // Store in our map and add to space
    bodyPartToShapes[bodyPart->getId()].push_back(shape);
    cpSpaceAddShape(space, shape);
    
    return shape;
}

void* ChipmunkPhysicsEngine::CreatePhysicsJoint(const Joint* joint) {
    if (!joint || !space) return nullptr;
    
    // Get the body parts that need to be connected
    BodyPart* bodyA = joint->getBodyA();
    BodyPart* bodyB = joint->getBodyB();
    
    if (!bodyA || !bodyB) return nullptr;
    
    // Get their root parents (these will have the actual physics bodies)
    BodyPart* rootA = bodyA->getRootParent();
    BodyPart* rootB = bodyB->getRootParent();
    
    // Find the corresponding physics bodies
    auto itA = bodyPartToBody.find(rootA->getId());
    auto itB = bodyPartToBody.find(rootB->getId());
    
    if (itA == bodyPartToBody.end() || itB == bodyPartToBody.end()) {
        return nullptr;
    }
    
    cpBody* cpBodyA = itA->second;
    cpBody* cpBodyB = itB->second;
    
    // Get the anchor points
    Vector2 anchorA = joint->getAnchorA();
    Vector2 anchorB = joint->getAnchorB();
    
    // Create the pivot joint
    // NOTE: Chipmunk's pivot joint takes world-space coordinates, so convert from local space
    cpVect worldAnchorA = cpBodyLocalToWorld(cpBodyA, cpv(anchorA.x, anchorA.y));
    cpVect worldAnchorB = cpBodyLocalToWorld(cpBodyB, cpv(anchorB.x, anchorB.y));
    
    // Create a pin joint (similar to a pivot joint but with two anchor points)
    cpConstraint* constraint = cpPinJointNew(cpBodyA, cpBodyB, 
                                            cpv(anchorA.x, anchorA.y), 
                                            cpv(anchorB.x, anchorB.y));
    
    // Set joint properties
    float stiffness = joint->getStiffness();
    if (stiffness > 0) {
        cpConstraintSetMaxForce(constraint, stiffness * 1000.0f);
    }
    
    // Handle collision between connected bodies
    cpConstraintSetCollideBodies(constraint, joint->getCollideConnected() ? cpTrue : cpFalse);
    
    // Store in our map and add to space
    jointToConstraint[joint->getId()] = constraint;
    cpSpaceAddConstraint(space, constraint);
    
    return constraint;
}

void ChipmunkPhysicsEngine::ProcessBodyPartHierarchy(const BodyPart* part, cpBody* parentBody) {
    if (!part || !parentBody) return;
    
    // Create shapes for this part and attach to parent body
    CreatePhysicsShape(part, parentBody);
    
    // Process all children recursively
    for (const BodyPart* child : part->getChildren()) {
        ProcessBodyPartHierarchy(child, parentBody);
    }
}

void ChipmunkPhysicsEngine::AddCreature(const Creature* creature) {
    if (!creature || !space) return;
    
    std::vector<void*> creatureObjects;
    
    // Get all body parts
    std::vector<BodyPart*> allParts = creature->getAllBodyParts();
    
    // First pass: create bodies for all root parts
    for (BodyPart* part : allParts) {
        // Only process root parts (no parent)
        if (part->getParent() == nullptr) {
            // Create physics body
            cpBody* body = static_cast<cpBody*>(CreatePhysicsBody(part));
            
            // Create shape for the root part
            CreatePhysicsShape(part, body);
            
            // Process children (they'll share the same body)
            for (BodyPart* child : part->getChildren()) {
                ProcessBodyPartHierarchy(child, body);
            }
            
            creatureObjects.push_back(body);
        }
    }
    
    // Second pass: create joints between parts
    for (const Joint* joint : creature->getJoints()) {
        void* constraint = CreatePhysicsJoint(joint);
        if (constraint) {
            creatureObjects.push_back(constraint);
        }
    }
    
    // Store all created objects
    creatureToObjects[creature->getId()] = creatureObjects;
}

void ChipmunkPhysicsEngine::RemoveCreature(const Creature* creature) {
    if (!creature || !space) return;
    
    unsigned creatureId = creature->getId();
    auto it = creatureToObjects.find(creatureId);
    
    if (it == creatureToObjects.end()) return;
    
    // Get all body parts and joints
    std::vector<BodyPart*> allParts = creature->getAllBodyParts();
    const std::vector<Joint*>& joints = creature->getJoints();
    
    // Remove joints first
    for (const Joint* joint : joints) {
        unsigned jointId = joint->getId();
        auto jIt = jointToConstraint.find(jointId);
        
        if (jIt != jointToConstraint.end()) {
            cpSpaceRemoveConstraint(space, jIt->second);
            cpConstraintFree(jIt->second);
            jointToConstraint.erase(jIt);
        }
    }
    
    // Then remove shapes and bodies
    for (BodyPart* part : allParts) {
        unsigned partId = part->getId();
        
        // Remove shapes
        auto sIt = bodyPartToShapes.find(partId);
        if (sIt != bodyPartToShapes.end()) {
            for (cpShape* shape : sIt->second) {
                cpSpaceRemoveShape(space, shape);
                cpShapeFree(shape);
            }
            bodyPartToShapes.erase(sIt);
        }
        
        // Remove body (only if it's a root part)
        if (part->getParent() == nullptr) {
            auto bIt = bodyPartToBody.find(partId);
            if (bIt != bodyPartToBody.end()) {
                cpSpaceRemoveBody(space, bIt->second);
                cpBodyFree(bIt->second);
                bodyPartToBody.erase(bIt);
            }
        }
    }
    
    // Remove from creature map
    creatureToObjects.erase(it);
}

Vector2 ChipmunkPhysicsEngine::GetBodyPosition(void* body) const {
    if (!body) return Vector2();
    
    cpVect pos = cpBodyGetPosition(static_cast<cpBody*>(body));
    return Vector2(pos.x, pos.y);
}

float ChipmunkPhysicsEngine::GetBodyRotation(void* body) const {
    if (!body) return 0.0f;
    
    return cpBodyGetAngle(static_cast<cpBody*>(body));
}

void ChipmunkPhysicsEngine::SetGravity(Vector2 gravity) {
    if (space) {
        cpSpaceSetGravity(space, cpv(gravity.x, gravity.y));
    }
}

Vector2 ChipmunkPhysicsEngine::GetGravity() const {
    if (space) {
        cpVect gravity = cpSpaceGetGravity(space);
        return Vector2(gravity.x, gravity.y);
    }
    return Vector2();
}

cpSpace* ChipmunkPhysicsEngine::GetNativeSpace() const {
    return space;
}
