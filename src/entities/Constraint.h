#ifndef CONSTRAINT_H
#define CONSTRAINT_H
#include "BodyPart.h"
#include <iostream>

enum ConstraintType {
    JOINT,
    MUSCLE
};

class Constraint {
private:
    unsigned id;
    BodyPart* partA;
    BodyPart* partB;
    Vector2 anchorA;
    Vector2 anchorB;
    float rest;
    float stiffness;
    float damping;
    bool collideConnected;

    ConstraintType type;
    
    static unsigned last_id;

public:
    Constraint(BodyPart* partA, BodyPart* partB, ConstraintType type,
        Vector2 anchorA = Vector2{}, Vector2 anchorB = Vector2{},
        float rest = 0.0f, float stiffness = 0, 
        float damping = 0, bool collideConnected = false);
    Constraint(BodyPart* partA, BodyPart* partB, ConstraintType type,
        Vector2 anchorA = Vector2{},
        float rest = 0.0f, float stiffness = 8e5, 
        float damping = 4e4, bool collideConnected = false);
    Constraint(const Constraint& other);
    ~Constraint();

    unsigned getId() const { return id; }
    BodyPart* getPartA() const { return partA; }
    BodyPart* getPartB() const { return partB; }
    Vector2 getAnchorA() const { return anchorA; }
    Vector2 getAnchorB() const { return anchorB; }
    float getRest() const { return rest; }
    float getStiffness() const { return stiffness; }
    float getDamping() const { return damping; }
    bool getCollideConnected() const { return collideConnected; }
    ConstraintType getType() const { return type; }

    void setBodyA(BodyPart* part) { partA = part; }
    void setBodyB(BodyPart* part) { partB = part; }
    void setAnchorA(Vector2 anchor) { anchorA = anchor; }
    void setAnchorB(Vector2 anchor) { anchorB = anchor; }
    void setRest(float value) { rest = value; }
    void setStiffness(float value) { stiffness = value; }
    void setDamping(float value) { damping = value; }
    void setCollideConnected(bool value) { collideConnected = value; }
    void setType(ConstraintType value) { type = value; }

    static unsigned newId();
    static void resetId();
};

#endif
