#include "ChipmunkEngine.h"

ChipmunkEngine::ChipmunkEngine() : space(nullptr) {}

ChipmunkEngine::~ChipmunkEngine() {
    if (space != nullptr) {
        shutdown();
    }
}

void ChipmunkEngine::initialize() {
    space = cpSpaceNew();
    cpSpaceSetGravity(space, cpv(0, 981));
    
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

cpShape* createShape(cpBody* body, const BodyPart *bodyPart) {
    const std::vector<Vector2>& vertices = bodyPart->getVertices();
    cpVect* cpVertices = new cpVect[vertices.size()];
    for (size_t i = 0; i < vertices.size(); ++i) {
        cpVertices[i] = cpv(vertices[i].x, vertices[i].y);
    }

    cpShape* shape = cpPolyShapeNew(body, vertices.size(), cpVertices, cpTransformIdentity, 0);

    cpShapeSetFriction(shape, bodyPart->getFriction());
    cpShapeSetElasticity(shape, bodyPart->getElasticity());
    cpShapeSetDensity(shape, bodyPart->getDensity());
    cpShapeSetUserData(shape, (void*)bodyPart);

    return shape;
}

void ChipmunkEngine::addBodyPart(unsigned creature_id, BodyPart *bodyPart)
{
    const float mass = bodyPart->getMass();
    cpFloat moment = cpMomentForPoly(
        mass,
        bodyPart->getVertices().size(),
        reinterpret_cast<const cpVect*>(bodyPart->getVertices().data()),
        cpvzero,
        0
    );
    cpBody* cp_body = cpBodyNew(mass, moment);
    cpBodySetUserData(cp_body, (void*)bodyPart);
    cpBodySetPosition(cp_body, cpv(0, 0));
    cpSpaceAddBody(space, cp_body);
    this->creatures[creature_id]->bodies[bodyPart->getId()] = cp_body;

    // Add shape
    cpShape* shape = createShape(cp_body, bodyPart);
    cpSpaceAddShape(space, shape);
    this->creatures[creature_id]->shapes[bodyPart->getId()] = shape;

    // Adding shapes if bodyPart has children
    for (auto& child : bodyPart->getAllChildren()) {
        cpShape* shape = createShape(cp_body, child);
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
    std::unordered_map<unsigned, BodyObject*> bodyMap;
    for (auto& creature : creatures) {
        for (auto& bodyPair : creature.second->bodies) {
            // Body
            BodyObject obj_body;
            cpBody* body = bodyPair.second;
            cpVect position = cpBodyGetPosition(body);
            cpVect velocity = cpBodyGetVelocity(body);

            obj_body.position.x = position.x;
            obj_body.position.y = position.y;

            obj_body.velocity.x = velocity.x;
            obj_body.velocity.y = velocity.y;

            obj_body.angle = cpBodyGetAngle(body);
            obj_body.mass = cpBodyGetMass(body);
            obj_body.id = bodyPair.first;
            bodies.push_back(obj_body);

            bodyMap[obj_body.id] = &bodies.back();
        }
        for (auto& shapePair : creature.second->shapes) {
            // Shape
            ShapeObject obj_shape;
            cpShape* shape = shapePair.second;
            cpBody* body = cpShapeGetBody(shape);

            unsigned id = shapePair.first;
            obj_shape.id = id;
            obj_shape.radius = creature.second->creature->getBodyPartById(id)->getRadius();
            obj_shape.vertices = creature.second->creature->getBodyPartById(id)->getVertices();

            BodyPart* bodyPartPtr = static_cast<BodyPart*>(cpShapeGetUserData(shape));
            if (id!=bodyPartPtr->getId()) {
                throw std::runtime_error("ChipmunkEngine::getRenderObjects: id does not match BodyPart id");
            }
            if (bodyMap.find(id) != bodyMap.end()) {
                obj_shape.body = bodyMap[id];
            } else {
                obj_shape.body = nullptr;
            }
            shapes.push_back(obj_shape);
        }
        for (auto& constraint : creature.second->constraints) {
            // Constraint
        }
    }
}
