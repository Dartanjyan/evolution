#include "PhysicsObjects.h"

BodyObject::BodyObject(unsigned id, Vector2 position, float angle, float mass, Vector2 velocity) : id(id), position(position), angle(angle), mass(mass), velocity(velocity)
{
}

ShapeObject::ShapeObject(
    unsigned id, 
    float radius, 
    std::vector<Vector2> vertices, 
    BodyObject* body,
    bool isWorldObj) : 
    id(id), 
    radius(radius), 
    vertices(vertices), 
    body(body),
    isWorldObj(isWorldObj)
{
    this->initShapeType();
}

void ShapeObject::initShapeType()
{
    size_t size = vertices.size();
    shapeType = size == 1 ? ShapeType::Circle : size == 2 ? ShapeType::Segment : ShapeType::Polygon;
}
