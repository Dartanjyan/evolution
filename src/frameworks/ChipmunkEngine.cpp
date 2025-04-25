#include "ChipmunkEngine.h"

#define CATEGORY_ENTITY  0b0001
#define CATEGORY_TERRAIN 0b0010

ChipmunkEngine::ChipmunkEngine() : space(nullptr) {}

ChipmunkEngine::~ChipmunkEngine() {
    if (space != nullptr) {
        shutdown();
    }
}

void ChipmunkEngine::initialize() {
    space = cpSpaceNew();
    cpSpaceSetGravity(space, cpv(0, 981));
    cpSpaceSetSleepTimeThreshold(space, 0.5);

    // Creating terrain
    cpFloat x = 1000;
    cpFloat y = 450;
    cpVect a = cpv(-x, y);
    cpVect b = cpv(x, y);
    cpBody* body = cpSpaceGetStaticBody(space);
    cpShape* terrain = cpSegmentShapeNew(body, a, b, 5.0);
    cpShapeSetFriction(terrain, 0.8);
    cpShapeSetElasticity(terrain, 0.5);
    cpShapeSetFilter(terrain, cpShapeFilterNew(0, CATEGORY_TERRAIN, CATEGORY_ENTITY));
    this->world_shapes.push_back(terrain);
    cpSpaceAddShape(space, terrain);
}

void ChipmunkEngine::update(float dt) {
    const int STEPS = 1;
    float sub_dt = dt / STEPS;
    step_mutex.lock();
    for (int i = 0; i < STEPS; ++i) {
        cpSpaceStep(space, sub_dt);
    }
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

cpShape* createShapeForBodyPart(cpBody* body, const BodyPart *bodyPart, cpVect bias) {
    const std::vector<Vector2>& vertices = bodyPart->getVertices();
    std::unique_ptr<cpVect[]> cpVertices = std::make_unique<cpVect[]>(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        cpVertices[i] = cpv(vertices[i].x, vertices[i].y) + bias;
    }

    cpShape* shape = nullptr;
    switch (vertices.size()) {
	case 0: {
	    throw std::runtime_error("DrawPanel::OnDraw(): shape must have at least 1 vertex\n");
	    break;
	}
	case 1: {
	    shape = cpCircleShapeNew(body, bodyPart->getRadius(), cpVertices[0]);
	    break;
	}
	case 2: {
            shape = cpSegmentShapeNew(body, cpVertices[0], cpVertices[1], bodyPart->getRadius());
	    break;
        }
        default: {
            shape = cpPolyShapeNew(body, vertices.size(), cpVertices.get(), cpTransformIdentity, 0);
	    break;
        }
    }

    cpShapeSetFriction(shape, bodyPart->getFriction());
    cpShapeSetElasticity(shape, bodyPart->getElasticity());
    cpShapeSetDensity(shape, bodyPart->getDensity());
    cpShapeSetUserData(shape, (void*)bodyPart);
    cpShapeSetFilter(shape, cpShapeFilterNew(0, CATEGORY_ENTITY, CATEGORY_TERRAIN));

    return shape;
}

void ChipmunkEngine::addShapeToChipmunkCreature(cpBody* body, BodyPart* bodyPart, unsigned creature_id, cpVect bias) {
    cpShape* shape = createShapeForBodyPart(body, bodyPart, bias);
    cpSpaceAddShape(space, shape);
    this->creatures[creature_id]->shapes[bodyPart->getId()] = shape;
    bodyPart->setBodyPosBias(Vector2(bias.x, bias.y));
}

void ChipmunkEngine::addBodyPart(unsigned creature_id, BodyPart *bodyPart)
{
    const float mass = bodyPart->getMass();
    auto vertices = bodyPart->getVertices();
    size_t vertices_count = vertices.size();
    cpFloat moment = 0;
    switch (vertices_count) {
        case 0:
            throw std::runtime_error("BodyPart must have at least 1 vertex");
            break;
        case 1:
            moment = cpMomentForCircle(
                mass,
                0,
                bodyPart->getRadius(),
                cpvzero
            );
            break;
        case 2:
            moment = cpMomentForSegment(
                mass,
                cpv(vertices[0].x, vertices[0].y),
                cpv(vertices[1].x, vertices[1].y),
                bodyPart->getRadius()
            );
            break;
        default:
            moment = cpMomentForPoly(
                mass,
                vertices_count,
                reinterpret_cast<const cpVect*>(vertices.data()),
                cpvzero,
                0
            );
            break;
    }
    Vector2 center = bodyPart->getCenter();
    cpVect body_pos = cpv(center.x, center.y);
    cpVect bias = -body_pos;

    cpBody* cp_body = cpBodyNew(mass, moment);
    cpBodySetUserData(cp_body, (void*)bodyPart);
    cpBodySetPosition(cp_body, body_pos);
    cpSpaceAddBody(space, cp_body);
    this->creatures[creature_id]->bodies[bodyPart->getId()] = cp_body;

    // Add shape
    this->addShapeToChipmunkCreature(cp_body, bodyPart, creature_id, bias);

    // Adding shapes if bodyPart has children
    for (auto& child : bodyPart->getAllChildren()) {
        this->addShapeToChipmunkCreature(cp_body, child, creature_id, bias);
    }
}

void ChipmunkEngine::addConstraint(unsigned creature_id, Constraint *constraint)
{
    cpBody* bodyA = this->creatures[creature_id]->bodies[constraint->getPartA()->getId()];
    cpBody* bodyB = this->creatures[creature_id]->bodies[constraint->getPartB()->getId()];

    cpVect anchorA = cpv(constraint->getAnchorA().x, constraint->getAnchorA().y);
    cpVect anchorB = cpv(constraint->getAnchorB().x, constraint->getAnchorB().y);

    cpConstraint* joint = nullptr;
    switch (constraint->getType()) {
        case ConstraintType::JOINT:
            joint = cpPivotJointNew(bodyA, bodyB, anchorA);
            break;
        case ConstraintType::MUSCLE:
            joint = cpDampedSpringNew(
                bodyA, bodyB, 
                anchorA, anchorB,
                constraint->getRest(), 
                constraint->getStiffness(), 
                constraint->getDamping()
            );
            break;
    }
    
    cpSpaceAddConstraint(space, joint);
    cpConstraintSetCollideBodies(joint, constraint->getCollideConnected());
    this->creatures[creature_id]->constraints[constraint->getId()] = joint;
}

void ChipmunkEngine::addCreature(Creature *creature)
{
    std::lock_guard<std::mutex> lock(data_mutex);

    ChimpmunkCreature* chimpmunkCreature = new ChimpmunkCreature();
    chimpmunkCreature->creature = creature;
    creatures[creature->getId()] = chimpmunkCreature;

    for (auto& bodyPart : creature->getMainBodyParts()) {
        addBodyPart(creature->getId(), bodyPart);
    }
    for (auto& constraint : creature->getConstraints()) {
        // I don't need constraints for now
        // break;
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
    // TODO: Implement removing creatures
}

void ChipmunkEngine::getRenderObjects(std::vector<BodyObject> &bodies, 
    std::vector<ShapeObject> &shapes, 
    std::vector<ConstraintObject> &constraints)
{
    std::lock_guard<std::mutex> lock(data_mutex);

    std::map<unsigned, size_t> bodyMap;

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
	        // Map body ids to their index at bodies vector
            bodyMap[obj_body.id] = bodies.size() - 1;

            // std::cout << "Body id: " << obj_body.id << ", position: " << obj_body.position << std::endl;
        }

        // std::cout<<"\n";

        for (auto& shapePair : creature.second->shapes) {
            ShapeObject obj_shape;
            // cpShape* shape = shapePair.second;
            unsigned id = shapePair.first;

            obj_shape.id = id;
            obj_shape.radius = creature.second->creature->getBodyPartById(id)->getRadius();
            obj_shape.vertices = creature.second->creature->getBodyPartById(id)->getBiasedVertices();

            auto it = bodyMap.find(id);
            if (it != bodyMap.end() && it->second < bodies.size()) {
                obj_shape.body = &bodies[it->second];
            } else {
                throw std::runtime_error("Body not found for shape. One of the condition is not met: " + 
                    std::to_string(it != bodyMap.end()) + 
                    " && " + 
                    std::to_string(it->second < bodies.size()) +
                    "\n"
                );
            }
            
            shapes.push_back(obj_shape);
        }

        // std::cout<<"\n";
    }

    BodyObject terrain_body;
    terrain_body.angle = 0;
    terrain_body.position = Vector2(0, 0);
    terrain_body.id= 0;
    bodies.push_back(terrain_body);

    ShapeObject terrain_shape;
    terrain_shape.body = &bodies.back();
    cpBB terrBB = cpShapeGetBB(world_shapes[0]);
    float y = (terrBB.b+terrBB.t)/2;
    terrain_shape.vertices = {Vector2(terrBB.l, y), Vector2(terrBB.r, y)};
    terrain_shape.id = 228;
    terrain_shape.radius=20;
    shapes.push_back(terrain_shape);
}
