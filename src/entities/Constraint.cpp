#include "Constraint.h"

unsigned Constraint::last_id = 0;

Constraint::Constraint(BodyPart* partA, BodyPart* partB, ConstraintType type, Vector2 anchorA, Vector2 anchorB, float rest, float stiffness, float damping, bool collideConnected): 
    id(Constraint::newId()), 
    partA(partA),
    partB(partB),
    anchorA(anchorA),
    anchorB(anchorB),
    rest(rest), 
    stiffness(stiffness), 
    damping(damping), 
    collideConnected(collideConnected),
    type(type)
{
    neutralRest = (anchorA-anchorB).length();
    float k = 20;
    minRest = neutralRest * (1-k);
    maxRest = neutralRest * (1+k);
    // std::cout << "NeutralRest: " << neutralRest << ", min: " << minRest << ", max: " << maxRest << "\n";
}

Constraint::Constraint(BodyPart* partA, BodyPart* partB, ConstraintType type, Vector2 anchorA, float rest, float stiffness, float damping, bool collideConnected): 
    id(Constraint::newId()), 
    partA(partA),
    partB(partB),
    anchorA(anchorA),
    anchorB(anchorA),
    rest(rest), 
    stiffness(stiffness), 
    damping(damping), 
    collideConnected(collideConnected),
    type(type)
{
}

Constraint::~Constraint() {
    // std::cout << "Deleting Joint, id = " << id << "\n";
}
/*
Constraint::Constraint(const Constraint &other):
    id(Constraint::newId()),
    partA(other.partA),
    partB(other.partB),
    anchorA(other.anchorA),
    anchorB(other.anchorB),
    rest(other.rest),
    stiffness(other.stiffness),
    damping(other.damping),
    collideConnected(other.collideConnected),
    type(other.type)
{
    // std::cout << "Copying Joint, id "<<other.id<<"->"<<id<< "\n";
}
*/

float Constraint::clampRest(float rest) const
{
    float a = std::clamp(rest, minRest, maxRest);
    // char buf[256];
    // snprintf(buf, 256, "rest = %f \tminRest = %f \tneutralRest = %f \tmaxRest = %f \treturn %f\n", rest, minRest, neutralRest, maxRest, a);
    // // std::cout << "rest = " << rest << " \tminRest = " << minRest << " \tmaxRest = " << maxRest << " \treturn " << a << "\n";
    // std::cout << buf;
    return a;
}

unsigned Constraint::newId() { return ++Constraint::last_id; }

void Constraint::resetId() { Constraint::last_id = 0; }

