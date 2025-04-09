#ifndef JOINT_H
#define JOINT_H
#include "BodyPart.h"
#include <iostream>

class Joint {
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
    // Конструкторы и деструктор
    Joint(BodyPart* bodyA, BodyPart* bodyB,
          Vector2 anchorA = Vector2{}, Vector2 anchorB = Vector2{},
          float rest = 0.0f, float stiffness = 0.0f, 
          float damping = 0.0f, bool collideConnected = false);
    Joint(const Joint& other);
    ~Joint();

    // Геттеры
    unsigned getId() const { return id; }
    BodyPart* getBodyA() const { return bodyA; }
    BodyPart* getBodyB() const { return bodyB; }
    Vector2 getAnchorA() const { return anchorA; }
    Vector2 getAnchorB() const { return anchorB; }
    float getRest() const { return rest; }
    float getStiffness() const { return stiffness; }
    float getDamping() const { return damping; }
    bool getCollideConnected() const { return collideConnected; }

    // Сеттеры
    void setBodyA(BodyPart* body) { bodyA = body; }
    void setBodyB(BodyPart* body) { bodyB = body; }
    void setAnchorA(Vector2 anchor) { anchorA = anchor; }
    void setAnchorB(Vector2 anchor) { anchorB = anchor; }
    void setRest(float value) { rest = value; }
    void setStiffness(float value) { stiffness = value; }
    void setDamping(float value) { damping = value; }
    void setCollideConnected(bool value) { collideConnected = value; }

    // Генерация нового ID
    static unsigned newId();
    static void resetId();
};

#endif