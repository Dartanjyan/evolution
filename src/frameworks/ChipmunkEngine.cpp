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
    step_mutex.lock();
    cpSpaceStep(space, dt);
    step_mutex.unlock();
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
    for (auto& shape : world_shapes) {
        cpSpaceRemoveShape(space, shape);
        cpShapeFree(shape);
    }
    for (auto& body : world_bodies) {
        cpSpaceRemoveBody(space, body);
        cpBodyFree(body);
    }
    cpSpaceFree(space);
    space = nullptr;
}

cpShape* createShape(cpBody* body, const BodyPart *bodyPart, cpVect bias) {
    const std::vector<Vector2>& vertices = bodyPart->getVertices();
    std::unique_ptr<cpVect[]> cpVertices = std::make_unique<cpVect[]>(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        cpVertices[i] = cpv(vertices[i].x, vertices[i].y) + bias;
    }

    cpShape* shape = nullptr;
    if (vertices.size() == 2) {
        shape = cpSegmentShapeNew(body, cpVertices[0], cpVertices[1], bodyPart->getRadius());
    } else {
        shape = cpPolyShapeNew(body, vertices.size(), cpVertices.get(), cpTransformIdentity, 0);
    }

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
    auto center = bodyPart->getCenter();
    cpVect body_pos = cpv(center.x, center.y);

    cpBody* cp_body = cpBodyNew(mass, moment);
    cpBodySetUserData(cp_body, (void*)bodyPart);
    cpBodySetPosition(cp_body, body_pos);
    cpSpaceAddBody(space, cp_body);
    this->creatures[creature_id]->bodies[bodyPart->getId()] = cp_body;

    // Add shape
    cpShape* shape = createShape(cp_body, bodyPart, -body_pos);
    cpSpaceAddShape(space, shape);
    this->creatures[creature_id]->shapes[bodyPart->getId()] = shape;

    // Adding shapes if bodyPart has children
    for (auto& child : bodyPart->getAllChildren()) {
        cpShape* shape = createShape(cp_body, child, -body_pos);
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
        break;
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
    std::vector<ConstraintObject> &constraints)
{
    std::map<unsigned, size_t> bodyMap; // Храним индексы вместо указателей

    // step_mutex.lock();

    for (auto& creature : creatures) {
        for (auto& bodyPair : creature.second->bodies) {
            BodyObject obj_body;
            cpBody* body = bodyPair.second;

            cpVect position = cpBodyGetPosition(body);
            cpVect velocity = cpBodyGetVelocity(body);

            obj_body.position = Vector2(position.x, position.y);
            obj_body.velocity = Vector2(velocity.x, velocity.y);
            obj_body.angle = cpBodyGetAngle(body);
            obj_body.mass = cpBodyGetMass(body);
            obj_body.id = bodyPair.first;

            bodies.push_back(obj_body);
            bodyMap[obj_body.id] = bodies.size() - 1; // Сохраняем индекс

            std::cout << "Body id: " << obj_body.id << ", position: " << obj_body.position << std::endl;
            if(std::isnan(obj_body.position.x) || std::isnan(obj_body.position.y)) {
                // throw std::runtime_error("Nan position\n");
            }
        }

        std::cout<<"\n";

        for (auto& shapePair : creature.second->shapes) {
            ShapeObject obj_shape;
            cpShape* shape = shapePair.second;
            unsigned id = shapePair.first;

            obj_shape.id = id;
            obj_shape.radius = creature.second->creature->getBodyPartById(id)->getRadius();
            obj_shape.vertices = creature.second->creature->getBodyPartById(id)->getVertices();

            // Получаем индекс из bodyMap и находим соответствующий BodyObject
            auto it = bodyMap.find(id);
            if (it != bodyMap.end() && it->second < bodies.size()) {
                obj_shape.body = &bodies[it->second];
            } else {
                obj_shape.body = nullptr;
            }

            std::cout << "Shape id: " << obj_shape.id << ", position: " << obj_shape.body->position << std::endl;

            shapes.push_back(obj_shape);
        }

        std::cout<<"\n";
    }

    // step_mutex.unlock();
}
