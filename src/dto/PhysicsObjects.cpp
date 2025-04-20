#include "PhysicsObjects.h"

BodyObject::BodyObject(unsigned id, Vector2 position, float angle, float mass, Vector2 velocity) : id(id), position(position), angle(angle), mass(mass), velocity(velocity)
{
}

ShapeObject::ShapeObject(unsigned id, float radius, std::vector<Vector2> vertices, BodyObject* body) : id(id), radius(radius), vertices(vertices), body(body)
{
}
