#include "BodyPart.h"
#include <iostream>

BodyPart::BodyPart(unsigned int id = 0, float density = 1.0f, float friction = 0.5f, float mass = 1.0f, float elasticity = 0.5f, bool isRootPart = false, bool isSensorPart = false)
: id(id), density(density), friction(friction), mass(mass), elasticity(elasticity), isRootPart(isRootPart), isSensorPart(isSensorPart) {}


BodyPart::BodyPart(const BodyPart &other)
    : id(other.id), density(other.density), friction(other.friction),
      mass(other.mass), elasticity(other.elasticity),
      isRootPart(other.isRootPart), isSensorPart(other.isSensorPart)
{
    // Copy constructor implementation
    std::cout << "BodyPart copy constructor called" << std::endl;
}

BodyPart::~BodyPart()
{
    // Clean up resources if needed
    std::cout << "BodyPart destructor called" << std::endl;
}
