#include "Constraint.h"

unsigned Constraint::last_id = 0;

Constraint::Constraint(BodyPart* bodyA, BodyPart* bodyB, Vector2 anchorA, Vector2 anchorB, float rest, float stiffness, float damping, bool collideConnected): 
    id(Constraint::newId()), 
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

Constraint::~Constraint() {
    // std::cout << "Deleting Joint, id = " << id << "\n";
}

Constraint::Constraint(const Constraint &other): 
    id(Constraint::newId()), 
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

unsigned Constraint::newId() { return ++Constraint::last_id; }

void Constraint::resetId() { Constraint::last_id = 0; }

