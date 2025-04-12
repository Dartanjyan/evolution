#include "Joint.h"

unsigned Joint::last_id = 0;

Joint::Joint(BodyPart* bodyA, BodyPart* bodyB, Vector2 anchorA, Vector2 anchorB, float rest, float stiffness, float damping, bool collideConnected): 
    id(Joint::newId()), 
    bodyA(bodyA),
    bodyB(bodyB),
    anchorA(anchorA),
    anchorB(anchorB),
    rest(rest), 
    stiffness(stiffness), 
    damping(damping), 
    collideConnected(collideConnected)
{
    // std::cout << "Creating Joint, id = "<<id<< "\n";
}

Joint::~Joint() {
    // std::cout << "Deleting Joint, id = " << id << "\n";
}

Joint::Joint(const Joint &other): 
    id(Joint::newId()), 
    bodyA(other.bodyA), 
    bodyB(other.bodyB),
    anchorA(other.anchorA),
    anchorB(other.anchorB),
    rest(other.rest), 
    stiffness(other.stiffness), 
    damping(other.damping), 
    collideConnected(other.collideConnected) 
{
    // std::cout << "Copying Joint, id "<<other.id<<"->"<<id<< "\n";
}

unsigned Joint::newId() { return ++Joint::last_id; }

void Joint::resetId() { Joint::last_id = 0; }

