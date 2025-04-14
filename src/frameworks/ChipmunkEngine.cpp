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
    for (auto& creature : creatures) {
        for (auto& body : creature.second->bodies) {
            cpSpaceRemoveBody(space, body.second);
            cpBodyFree(body.second);
        }
        for (auto& shape : creature.second->shapes) {
            cpSpaceRemoveShape(space, shape.second);
            cpShapeFree(shape.second);
        }
        for (auto& constraint : creature.second->constraints) {
            cpSpaceRemoveConstraint(space, constraint.second);
            cpConstraintFree(constraint.second);
        }
        delete creature.second;
    }
    cpSpaceFree(space);
    space = nullptr;
}

cpShape* createShape(cpBody* body, const std::vector<Vector2>& vertices) {
    cpVect* cpVertices = new cpVect[vertices.size()];
    for (size_t i = 0; i < vertices.size(); ++i) {
        cpVertices[i] = cpv(vertices[i].x, vertices[i].y);
    }
    cpShape* shape = cpPolyShapeNew(body, vertices.size(), cpVertices, cpTransformIdentity, 0);
    return shape;
}

void ChipmunkEngine::addBodyPart(unsigned creature_id, BodyPart *bodyPart)
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
    this->creatures[creature_id]->bodies[bodyPart->getId()] = body;

    for (auto& child : bodyPart->getAllChildren()) {
        cpShape* shape = createShape(body, child->getVertices());
        cpShapeSetFriction(shape, child->getFriction());
        cpShapeSetElasticity(shape, child->getElasticity());
        cpSpaceAddShape(space, shape);
        this->creatures[creature_id]->shapes[child->getId()] = shape;
    }
}

void ChipmunkEngine::addConstraint(unsigned creature_id, Constraint *constraint)
{
    cpBody* bodyA = this->creatures[creature_id]->bodies[constraint->getPartA()->getId()];
    cpBody* bodyB = this->creatures[creature_id]->bodies[constraint->getPartB()->getId()];

    cpVect anchorA = cpv(constraint->getAnchorA().x, constraint->getAnchorA().y);
    cpVect anchorB = cpv(constraint->getAnchorB().x, constraint->getAnchorB().y);

    cpConstraint* joint = nullptr;
    if (constraint->getType() == ConstraintType::JOINT) {
        joint = cpPivotJointNew(bodyA, bodyB, anchorA);
    } else {
        joint = cpDampedSpringNew(bodyA, bodyB, anchorA, anchorB,
            constraint->getRest(), constraint->getStiffness(), constraint->getDamping());
    }
    cpSpaceAddConstraint(space, joint);
    cpConstraintSetCollideBodies(joint, constraint->getCollideConnected());
    this->creatures[creature_id]->constraints[constraint->getId()] = joint;
}

void ChipmunkEngine::addCreature(Creature *creature)
{
    ChimpmunkCreature* chimpmunkCreature = new ChimpmunkCreature();
    chimpmunkCreature->creature = creature;
    creatures[creature->getId()] = chimpmunkCreature;

    for (auto& bodyPart : creature->getMainBodyParts()) {
        addBodyPart(creature->getId(), bodyPart);
    }
    for (auto& constraint : creature->getConstraints()) {
        addConstraint(creature->getId(), constraint);
    }
}

void ChipmunkEngine::removeBodyPart(unsigned creature_id, BodyPart *bodyPart)
{
    auto it = creatures.find(creature_id);
    if (it != creatures.end()) {
        ChimpmunkCreature* chimpmunkCreature = it->second;
        auto bodyIt = chimpmunkCreature->bodies.find(bodyPart->getId());
        if (bodyIt != chimpmunkCreature->bodies.end()) {
            cpBody* body = bodyIt->second;
            cpSpaceRemoveBody(space, body);
            cpBodyFree(body);
            chimpmunkCreature->bodies.erase(bodyIt);
        }
    }
}

void ChipmunkEngine::removeConstraint(unsigned creature_id, Constraint *constraint)
{
    auto it = creatures.find(creature_id);
    if (it != creatures.end()) {
        ChimpmunkCreature* chimpmunkCreature = it->second;
        auto constraintIt = chimpmunkCreature->constraints.find(constraint->getId());
        if (constraintIt != chimpmunkCreature->constraints.end()) {
            cpConstraint* constraint = constraintIt->second;
            cpSpaceRemoveConstraint(space, constraint);
            cpConstraintFree(constraint);
            chimpmunkCreature->constraints.erase(constraintIt);
        }
    }
}

void ChipmunkEngine::removeCreature(unsigned creature_id)
{
    Creature* creature = creatures[creature_id]->creature;
    if (creature == nullptr) {
        return;
    }
}

void ChipmunkEngine::getRenderObjects(std::vector<BodyObject> &bodies, 
    std::vector<ShapeObject> &shapes, 
    std::vector<ConstraintObject> &constraints) const
{
    // TODO: Implement this function to fill the bodies, shapes, and constraints vectors
    for (const auto& creature : creatures) {

    }
}
