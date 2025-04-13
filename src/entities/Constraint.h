#ifndef CONSTRAINT_H
#define CONSTRAINT_H
#include "BodyPart.h"
#include <iostream>

class Constraint {
private:
    unsigned id;
    BodyPart* bodyA;
    BodyPart* bodyB;
    Vector2 anchorA;
    Vector2 anchorB;
    float rest;
    float stiffness;
    float damping;
    bool collideConnected;
    
    static unsigned last_id;

public:
    Constraint(BodyPart* bodyA, BodyPart* bodyB,
          Vector2 anchorA = Vector2{}, Vector2 anchorB = Vector2{},
          float rest = 0.0f, float stiffness = 0.0f, 
          float damping = 0.0f, bool collideConnected = false);
    Constraint(const Constraint& other);
    ~Constraint();

    unsigned getId() const { return id; }
    BodyPart* getBodyA() const { return bodyA; }
    BodyPart* getBodyB() const { return bodyB; }
    Vector2 getAnchorA() const { return anchorA; }
    Vector2 getAnchorB() const { return anchorB; }
    float getRest() const { return rest; }
    float getStiffness() const { return stiffness; }
    float getDamping() const { return damping; }
    bool getCollideConnected() const { return collideConnected; }

    void setBodyA(BodyPart* body) { bodyA = body; }
    void setBodyB(BodyPart* body) { bodyB = body; }
    void setAnchorA(Vector2 anchor) { anchorA = anchor; }
    void setAnchorB(Vector2 anchor) { anchorB = anchor; }
    void setRest(float value) { rest = value; }
    void setStiffness(float value) { stiffness = value; }
    void setDamping(float value) { damping = value; }
    void setCollideConnected(bool value) { collideConnected = value; }

    static unsigned newId();
    static void resetId();
};

enum JointTypes
{

};

#endif
