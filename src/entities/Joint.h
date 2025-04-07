#ifndef JOINT_H
#define JOINT_H
#include "BodyPart.h"

class Joint {
private:
    unsigned id;
    BodyPart* bodyA;
    BodyPart* bodyB;
    // anchorA, B
    float rest;
    float stiffness;
    float damping;
    bool collideConnected;
public:
    Joint(unsigned id,
        BodyPart* bodyA,
        BodyPart* bodyB,
        //anchorA, B
        float rest,
        float stiffness,
        float damping,
        bool collideConnected);
    ~Joint();
    Joint(const Joint& other);

    unsigned getId() const;
    BodyPart* getBodyA() const;
    BodyPart* getBodyB() const;

    float getRest() const;
    float getStiffness() const;
    float getDamping() const;
    bool getCollideConnected() const;
};

#endif