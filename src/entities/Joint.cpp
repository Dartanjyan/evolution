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
    collideConnected(collideConnected) {}

Joint::~Joint() {}

Joint::Joint(const Joint &other): 
    id(Joint::newId()), 
    bodyA(other.bodyA), 
    bodyB(other.bodyB),
    anchorA(other.anchorA),
    anchorB(other.anchorB),
    rest(other.rest), 
    stiffness(other.stiffness), 
    damping(other.damping), 
    collideConnected(other.collideConnected) {}

unsigned Joint::newId() { return Joint::last_id++; }

void Joint::resetId() { Joint::last_id = 0; }

