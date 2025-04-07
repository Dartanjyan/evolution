#include "Joint.h"

Joint::Joint(unsigned id, BodyPart *bodyA, BodyPart *bodyB, float rest, float stiffness, float damping, bool collideConnected) : id(id), bodyA(bodyA), bodyB(bodyB), rest(rest), stiffness(stiffness), damping(damping), collideConnected(collideConnected)
{
}
Joint::~Joint() 
{

}
Joint::Joint(const Joint &other) : id(other.id), bodyA(other.bodyA), bodyB(other.bodyB), rest(other.rest), stiffness(other.stiffness), damping(other.damping), collideConnected(other.collideConnected) 
{

}
