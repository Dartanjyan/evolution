#include "ChipmunkEngine.h"

ChipmunkEngine::ChipmunkEngine() : space(nullptr) {}

ChipmunkEngine::~ChipmunkEngine() {
    shutdown();
}

void ChipmunkEngine::initialize() {
    space = cpSpaceNew();
    cpSpaceSetGravity(space, cpv(0, -100));
    // Here I may add more settings
}

void ChipmunkEngine::update(float dt) {
    cpSpaceStep(space, dt);
}

void ChipmunkEngine::shutdown() {
    for (auto& body : bodies) {
        cpSpaceRemoveBody(space, body);
        cpBodyFree(body);
    }
    bodies.clear();
    for (auto& shape : shapes) {
        cpSpaceRemoveShape(space, shape);
        cpShapeFree(shape);
    }
    shapes.clear();
    for (auto& constraint : constraints) {
        cpSpaceRemoveConstraint(space, constraint);
        cpConstraintFree(constraint);
    }
    constraints.clear();
    for (auto& creature : creatures) {
        delete creature;
    }
    creatures.clear();
    if (space) {
        cpSpaceFree(space);
        space = nullptr;
    }
}

cpShape* createShape(cpBody* body, const std::vector<Vector2>& vertices) {
    cpVect* cpVertices = new cpVect[vertices.size()];
    for (size_t i = 0; i < vertices.size(); ++i) {
        cpVertices[i] = cpv(vertices[i].x, vertices[i].y);
    }
    cpShape* shape = cpPolyShapeNew(body, vertices.size(), cpVertices, cpTransformIdentity, 0);
    return shape;
}

void ChipmunkEngine::addBodyPart(BodyPart *bodyPart)
{
    float mass = bodyPart->getMass();
    cpFloat moment = cpMomentForPoly(
        mass,
        bodyPart->getVertices().size(),
        reinterpret_cast<const cpVect*>(bodyPart->getVertices().data()),
        cpvzero,
        0
    );
    cpBody* body = cpBodyNew(mass, moment);
    cpSpaceAddBody(space, body);
    this->bodies.push_back(body);

    for (auto& child : bodyPart->getAllChildren()) {
        cpShape* shape = createShape(body, child->getVertices());
        cpShapeSetFriction(shape, child->getFriction());
        cpShapeSetElasticity(shape, child->getElasticity());
        cpSpaceAddShape(space, shape);
        this->shapes.push_back(shape);
    }
}

void ChipmunkEngine::addConstraint(Constraint *constraint)
{
}

void ChipmunkEngine::addCreature(Creature *creature)
{
}

void ChipmunkEngine::removeBodyPart(BodyPart *bodyPart)
{
}

void ChipmunkEngine::removeConstraint(Constraint *constraint)
{
}

void ChipmunkEngine::removeCreature(Creature *creature)
{
}

void ChipmunkEngine::getRenderObjects(std::vector<BodyObject> &bodies, 
    std::vector<ShapeObject> &shapes, 
    std::vector<ConstraintObject> &joints) const
{
}
