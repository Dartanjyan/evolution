#include "ChipmunkEngine.h"
#include <memory>

#define CATEGORY_ENTITY  0b0001
#define CATEGORY_TERRAIN 0b0010

ChipmunkCreature* ChipmunkEngine::findChipmunkCreatureForCreature(Creature *creature)
{
    auto it = chipmunkCreatures.find(creature->getId());
    if (it != chipmunkCreatures.end()) {
        return it->second;
    } else {
        throw std::runtime_error("ChipmunkEngine::findChipmunkCreatureForCreature: Creature not found");
    }
}

ChipmunkEngine::ChipmunkEngine() : space(nullptr) {}

ChipmunkEngine::~ChipmunkEngine() {
    if (space != nullptr) {
        shutdown();
    }
}

void ChipmunkEngine::initialize() {
    space = cpSpaceNew();
    cpSpaceSetGravity(space, cpv(0, 981*1.5));
    cpSpaceSetSleepTimeThreshold(space, 0.5);

    // Amount of overlap between shapes that is allowed
    cpSpaceSetCollisionSlop(space, 0.8);

    // Chipmunk attempts to correct 10% of error ever 1/60th of a second
    cpSpaceSetCollisionBias(space, cpfpow(1.0f - 0.1f, 60.0f));

    // Creating terrain
    cpFloat x = 1000;
    cpFloat y = 350;
    cpVect a = cpv(-x, y);
    cpVect b = cpv(x, y);
    cpBody* body = cpSpaceGetStaticBody(space);
    cpShape* terrain = cpSegmentShapeNew(body, a, b, 5.0);
    cpShapeSetFriction(terrain, 0.8);
    cpShapeSetElasticity(terrain, 0.5);
    cpShapeSetFilter(terrain, cpShapeFilterNew(0, CATEGORY_TERRAIN, CATEGORY_ENTITY));
    this->world_shapes.push_back(terrain);
    cpSpaceAddShape(space, terrain);

    std::cout << "ChipmunkEngine initialized\n";
}

void ChipmunkEngine::update(float dt) {
    static const int STEPS = 1;
    static const float sub_dt = dt / STEPS;
    step_mutex.lock();
    for (int i = 0; i < STEPS; ++i) {
        cpSpaceStep(space, sub_dt);
    }
    step_mutex.unlock();
}

void ChipmunkEngine::removeCreature(unsigned creature_id)
{
    ChipmunkCreature* creature = chipmunkCreatures[creature_id];
    if (creature == nullptr) {
        return;
    }
    for (auto& body : creature->bodies) {
        cpSpaceRemoveBody(space, body.second);
        cpBodyFree(body.second);
    }
    for (auto& shape : creature->shapes) {
        cpSpaceRemoveShape(space, shape.second);
        cpShapeFree(shape.second);
    }
    for (auto& constraint : creature->constraints) {
        cpSpaceRemoveConstraint(space, constraint.second);
        cpConstraintFree(constraint.second);
    }
    delete creature->creature;
    delete creature;
    // std::cout << "Removed creature " << creature_id << "\n";
}

void ChipmunkEngine::shutdown() {
    std::vector<unsigned> creatureIds;
    for (const auto& pair : chipmunkCreatures) {
        creatureIds.push_back(pair.first);
    }
    for (unsigned id : creatureIds) {
        removeCreature(id);
    }
    chipmunkCreatures.clear();
    for (auto& shape : world_shapes) {
        cpSpaceRemoveShape(space, shape);
        cpShapeFree(shape);
    }
    world_shapes.clear();
    for (auto& body : world_bodies) {
        cpSpaceRemoveBody(space, body);
        cpBodyFree(body);
    }
    world_bodies.clear();
    cpSpaceFree(space);
    space = nullptr;

    std::cout << "ChipmunkEngine has been shut down.\n";
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
    this->chipmunkCreatures[creature_id]->shapes[bodyPart->getId()] = shape;
    bodyPart->setBodyPosBias(Vector2(bias.x, bias.y));
}

void ChipmunkEngine::addBodyPart(unsigned creature_id, BodyPart *bodyPart)
{
    const float mass = bodyPart->getMass();
    auto vertices = bodyPart->getVertices();
    size_t vertices_count = vertices.size();
    cpFloat moment = 0;
    cpVect cp_verts[vertices_count];
    for (size_t i=0; i<vertices_count; i++) {
        cp_verts[i].x = vertices[i].x;
        cp_verts[i].y = vertices[i].y;
    }
    switch (vertices_count) {
        case 0:
            throw std::runtime_error("BodyPart must have at least 1 vertex");
            break;
        case 1:
            moment = cpMomentForCircle(
                mass,
                0,
                bodyPart->getRadius(),
                cp_verts[0]
            );
            break;
        case 2:
            moment = cpMomentForSegment(
                mass,
                cp_verts[0],
                cp_verts[1],
                bodyPart->getRadius()
            );
            break;
        default:
            moment = cpMomentForPoly(
                mass,
                vertices_count,
                cp_verts,
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
    this->chipmunkCreatures[creature_id]->bodies[bodyPart->getId()] = cp_body;

    // Add shape
    this->addShapeToChipmunkCreature(cp_body, bodyPart, creature_id, bias);

    // Adding shapes if bodyPart has children
    for (auto& child : bodyPart->getAllChildren()) {
        this->addShapeToChipmunkCreature(cp_body, child, creature_id, bias);
    }
}

void ChipmunkEngine::addConstraint(unsigned creature_id, Constraint *constraint)
{
    cpBody* bodyA = this->chipmunkCreatures[creature_id]->bodies[constraint->getPartA()->getId()];
    cpBody* bodyB = this->chipmunkCreatures[creature_id]->bodies[constraint->getPartB()->getId()];

    cpVect anchorA = cpv(constraint->getAnchorA().x, constraint->getAnchorA().y);
    cpVect anchorB = cpv(constraint->getAnchorB().x, constraint->getAnchorB().y);

    cpConstraint* cp_constraint = nullptr;
    switch (constraint->getType()) {
        case ConstraintType::JOINT:
            cp_constraint = cpPivotJointNew(bodyA, bodyB, anchorA);
            break;
        case ConstraintType::MUSCLE:
            cp_constraint = cpDampedSpringNew(
                bodyA, bodyB, 
                anchorA, anchorB,
                constraint->getRest(), 
                constraint->getStiffness(), 
                constraint->getDamping()
            );
            break;
    }
    
    cpConstraintSetErrorBias(cp_constraint, cpfpow(1.0 - 0.01, 1200000));
    cpConstraintSetCollideBodies(cp_constraint, constraint->getCollideConnected());
    this->chipmunkCreatures[creature_id]->constraints[constraint->getId()] = cp_constraint;
    cpConstraintSetUserData(cp_constraint, (void*)constraint);
    cpSpaceAddConstraint(space, cp_constraint);
}

void ChipmunkEngine::addCreature(Creature *creature)
{
    std::lock_guard<std::mutex> lock(data_mutex);

    ChipmunkCreature* chimpmunkCreature = new ChipmunkCreature();
    chimpmunkCreature->creature = creature;
    chipmunkCreatures[creature->getId()] = chimpmunkCreature;

    for (auto& bodyPart : creature->getMainBodyParts()) {
        addBodyPart(creature->getId(), bodyPart);
    }
    for (auto& constraint : creature->getConstraints()) {
        // I don't need constraints for now
        // break;
        addConstraint(creature->getId(), constraint);
    }
}

// NOTE: Maybe not useful
void ChipmunkEngine::removeBodyPart(unsigned creature_id, BodyPart *bodyPart)
{
    auto it = chipmunkCreatures.find(creature_id);
    if (it != chipmunkCreatures.end()) {
        ChipmunkCreature* chimpmunkCreature = it->second;
        auto bodyIt = chimpmunkCreature->bodies.find(bodyPart->getId());
        if (bodyIt != chimpmunkCreature->bodies.end()) {
            cpBody* body = bodyIt->second;
            cpSpaceRemoveBody(space, body);
            cpBodyFree(body);
            chimpmunkCreature->bodies.erase(bodyIt);
        }
    }
}

// NOTE: Maybe not useful
void ChipmunkEngine::removeConstraint(unsigned creature_id, Constraint *constraint)
{
    auto it = chipmunkCreatures.find(creature_id);
    if (it != chipmunkCreatures.end()) {
        ChipmunkCreature* chimpmunkCreature = it->second;
        auto constraintIt = chimpmunkCreature->constraints.find(constraint->getId());
        if (constraintIt != chimpmunkCreature->constraints.end()) {
            cpConstraint* constraint = constraintIt->second;
            cpSpaceRemoveConstraint(space, constraint);
            cpConstraintFree(constraint);
            chimpmunkCreature->constraints.erase(constraintIt);
        }
    }
}

const size_t MAX_PROCESSED_CREATURES = 20;
void ChipmunkEngine::getRenderObjects(std::vector<BodyObject> &bodies, 
    std::vector<ShapeObject> &shapes, 
    std::vector<ConstraintObject> &constraints)
{
    std::lock_guard<std::mutex> lock(data_mutex);

    std::map<unsigned, size_t> bodyPartIdToBodiesId;
    
    std::map<unsigned int, ChipmunkCreature *> shownCreatures = chipmunkCreatures;

    size_t creatureCount = 0;
    for (auto& creature : chipmunkCreatures) {
        if (creatureCount++ >= MAX_PROCESSED_CREATURES) break;
        for (auto& bodyPair : creature.second->bodies) {
            BodyObject obj_body;
            cpBody* cp_body = bodyPair.second;
            BodyPart* bodyPart = static_cast<BodyPart*>(cpBodyGetUserData(cp_body));
            if (cp_body == nullptr || bodyPart == nullptr)
                continue;

            cpVect position = cpBodyGetPosition(cp_body);
            cpVect velocity = cpBodyGetVelocity(cp_body);

            obj_body.id = bodyPart->getId();
            obj_body.position = Vector2(position.x, position.y);
            obj_body.velocity = Vector2(velocity.x, velocity.y);
            obj_body.angle = cpBodyGetAngle(cp_body);
            obj_body.mass = cpBodyGetMass(cp_body);
            obj_body.initPosition = bodyPart->getBodyPosBias();

            bodies.push_back(obj_body);
            bodyPartIdToBodiesId[obj_body.id] = bodies.size() - 1;
        }
    }

    creatureCount = 0;
    for (auto& creature : chipmunkCreatures) {
        if (creatureCount++ >= MAX_PROCESSED_CREATURES) break;

        for (auto& shapePair : creature.second->shapes) {
            cpShape* cp_shape = shapePair.second;
            BodyPart* bodyPart = static_cast<BodyPart*>(cpShapeGetUserData(cp_shape));
            if (cp_shape == nullptr || bodyPart == nullptr)
                continue;

            ShapeObject obj_shape;
            obj_shape.id = shapePair.first;
            obj_shape.radius = bodyPart->getRadius();
            obj_shape.vertices = bodyPart->getBiasedVertices();

            auto it = bodyPartIdToBodiesId.find(bodyPart->getId());
            if (it != bodyPartIdToBodiesId.end()) {
                obj_shape.body = &bodies.at(it->second);
            } else {
                obj_shape.body = nullptr;
                std::cerr << "BodyObject not found for BodyPart ID: " << bodyPart->getId() << std::endl;
            }

            obj_shape.isWorldObj = false;
            obj_shape.initShapeType();
            shapes.push_back(obj_shape);
        }

    // }
    // creatureCount = 0;
    // for (auto& creature : chipmunkCreatures) {
    //     if (creatureCount++ >= MAX_PROCESSED_CREATURES) break;

        for (auto& constraintPair : creature.second->constraints) {
            cpConstraint* cp_constraint = constraintPair.second;
            ConstraintObject obj_constraint;
            Constraint* basic_constraint = static_cast<Constraint*>(cpConstraintGetUserData(cp_constraint));
            cpBody *cp_bodyA = cpConstraintGetBodyA(cp_constraint);
            cpBody *cp_bodyB = cpConstraintGetBodyB(cp_constraint);
            if (basic_constraint == nullptr || cp_bodyA == nullptr || cp_bodyB == nullptr)
                continue;

            obj_constraint.id = constraintPair.first;
            obj_constraint.constraintType = basic_constraint->getType();
            if (obj_constraint.constraintType == ConstraintType::MUSCLE) {
                
                // partA
                BodyPart* bodyPartA = static_cast<BodyPart*>(cpBodyGetUserData(cpConstraintGetBodyA(cp_constraint)));
                auto itA = bodyPartIdToBodiesId.find(bodyPartA->getId());
                if (itA != bodyPartIdToBodiesId.end()) {
                    obj_constraint.partA = &bodies.at(itA->second);
                } else {
                    obj_constraint.partA = nullptr;
                    std::cerr << "BodyObject not found for BodyPart ID: " << bodyPartA->getId() << std::endl;
                }

                // partB
                BodyPart* bodyPartB = static_cast<BodyPart*>(cpBodyGetUserData(cpConstraintGetBodyB(cp_constraint)));
                auto itB = bodyPartIdToBodiesId.find(bodyPartB->getId());
                if (itB != bodyPartIdToBodiesId.end()) {
                    obj_constraint.partB = &bodies.at(itB->second);
                } else {
                    obj_constraint.partB = nullptr;
                    std::cerr << "BodyObject not found for BodyPart ID: " << bodyPartB->getId() << std::endl;
                }

                cpVect anchorA = cpDampedSpringGetAnchorA(cp_constraint);
                cpVect anchorB = cpDampedSpringGetAnchorB(cp_constraint);
                obj_constraint.anchorA = Vector2(anchorA.x, anchorA.y) + obj_constraint.partA->initPosition;
                obj_constraint.anchorB = Vector2(anchorB.x, anchorB.y) + obj_constraint.partB->initPosition;

                constraints.push_back(obj_constraint);
            }
        }
    }

    BodyObject terrain_body;
    terrain_body.id= 0;
    terrain_body.angle = 0;
    terrain_body.position = Vector2(0, 0);
    bodies.push_back(terrain_body);

    ShapeObject terrain_shape;
    terrain_shape.body = &bodies.back();
    terrain_shape.id = 228;
    terrain_shape.radius=20;
    terrain_shape.isWorldObj = true;

    cpBB terrBB = cpShapeGetBB(world_shapes[0]);
    float y = (terrBB.b+terrBB.t)/2;
    terrain_shape.vertices = {Vector2(terrBB.l, y), Vector2(terrBB.r, y)};
    
    terrain_shape.initShapeType();
    shapes.push_back(terrain_shape);
}

void ChipmunkEngine::getCreatureAIInputs(std::vector<CreaturePhysicsInputs>& out) {
    // root part angle sin and cos
    // each joint's relative angle between two connected body parts - sin and cos
    // for each sight part: 5 raycasts
    // memory of the creature
    // NOT NOW: bool for every sensitive part: 1 if touches ground, else 0
    //
    // In result, length will be (without sensitive parts):
    // 2 + joints_amount*2 + eyes_amount*5
    out.clear();
    out.reserve(chipmunkCreatures.size());
    for (auto& creature_pair : chipmunkCreatures) {
        CreaturePhysicsInputs data;
        data.creature = creature_pair.second->creature;

        std::vector<Constraint*> joints = data.creature->getJoints();
        std::vector<BodyPart*> eyes = data.creature->getSightParts();
        std::vector<double> memory = data.creature->getBrain()->getMemory();
        int amount = 2 + joints.size()*2 + eyes.size()*5 + memory.size();
        data.inputs.reserve(amount);

        const float angle = cpBodyGetAngle((*creature_pair.second->bodies.begin()).second);
        data.inputs.emplace_back(std::sin(angle));
        data.inputs.emplace_back(std::cos(angle));
        for (auto jointPair : creature_pair.second->constraints) {
            cpConstraint* joint = jointPair.second;
            if (cpConstraintIsPivotJoint(joint)) {
                float relativeAngle = cpBodyGetAngle(cpConstraintGetBodyA(joint)) - cpBodyGetAngle(cpConstraintGetBodyB(joint));
                data.inputs.emplace_back(std::sin(relativeAngle));
                data.inputs.emplace_back(std::cos(relativeAngle));
            }
        }
        for (size_t i = 0; i < eyes.size()*5; i++) {
            // placeholder until eyesight is implemented
            data.inputs.emplace_back(0.0f);
        }
        data.inputs.insert(data.inputs.end(), memory.begin(), memory.end());
        out.push_back(data);
        
        // static bool outputted = false;
        // if (!outputted) {
        //     std::cout << "ChipmunkEngine::getCreatureAIInputs(): data has length of " << amount << std::endl;
        //     outputted = true;
        // }
    }
}

void ChipmunkEngine::getCreatures(std::vector<Creature *>& out)
{
    for (auto& creaturePair : chipmunkCreatures) {
        out.emplace_back(creaturePair.second->creature);
    }
}

void ChipmunkEngine::applyAIResults(const std::vector<CreaturePhysicsInputs> &data)
{
    const float MUSCLE_WORK_FITNESS_IMPACT = 0.001;
    const float X_DISTANCE_FITNESS_IMPACT = 0.001;

    for (const auto& d : data) {
        ChipmunkCreature *creature = findChipmunkCreatureForCreature(d.creature);
        auto muscles = creature->creature->getMuscles();
        if (d.outputs.size() != muscles.size()) {
            std::cout << "ChipmunkEngine::applyAIResults: Got incompatible output size to muscle amount (got "<<d.outputs.size()<<", expected "<<muscles.size()<<")\n";
            continue;
        }
        
        for (size_t i = 0; i < muscles.size(); i++) {
            Constraint* muscle = muscles[i];
            cpConstraint* chipmunkMuscle = creature->constraints[muscle->getId()];

            float old_rest = cpDampedSpringGetRestLength(chipmunkMuscle);
            float new_rest = d.outputs[i] * muscle->getNeutralSize() * 4;
            cpDampedSpringSetRestLength(chipmunkMuscle, new_rest);

            // A little penalty for every muscle work
            d.creature->setFitness(d.creature->getFitness() - std::abs(old_rest - new_rest)*MUSCLE_WORK_FITNESS_IMPACT);
            // std::cout << "New rest: " << d.outputs[i] << "\n";
            // std::cout << "Rest diff: " << old_rest - new_rest << "\n";
        }
        
        if (!creature->bodies.empty()) {
            auto firstBodyIt = creature->bodies.begin();
            cpBody* firstBody = firstBodyIt->second;
            auto cpPos = cpBodyGetPosition(firstBody);
            if (!creature->posInitialized) {
                creature->posInitialized = true;
                creature->lastPos = Vector2(cpPos.x, cpPos.y);
            } else {
                auto pos = Vector2(cpPos.x, cpPos.y);
                Vector2 distance = creature->lastPos - pos;
                creature->creature->setFitness(creature->creature->getFitness() + distance.x*X_DISTANCE_FITNESS_IMPACT);
                creature->lastPos = pos;
            }
        }
    }
}
