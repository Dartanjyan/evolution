#include "ChipmunkEngine.h"
#include <algorithm>
#include <memory>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <thread>

#define CATEGORY_ENTITY  0b0001
#define CATEGORY_TERRAIN 0b0010

namespace {
constexpr cpFloat TERRAIN_HALF_WIDTH = 3000;
constexpr cpFloat TERRAIN_Y = 350;
constexpr cpFloat TERRAIN_RADIUS = 5;
constexpr int SUBSTEPS = 3;
constexpr unsigned MAX_PHYSICS_THREADS = 8;

void stepSpaces(std::span<cpSpace* const> spaces, cpFloat dt) {
    for (cpSpace* space : spaces) {
        for (int i = 0; i < SUBSTEPS; ++i) {
            cpSpaceStep(space, dt);
        }
    }
}
}

ChipmunkCreature::ChipmunkCreature() : space(cpSpaceNew()) {
    cpSpaceSetGravity(space, cpv(0, 981*1.5));
    cpSpaceSetSleepTimeThreshold(space, 0.5);

    // Amount of overlap between shapes that is allowed
    cpSpaceSetCollisionSlop(space, 0);

    // Chipmunk attempts to correct 10% of error ever 1/60th of a second
    // cpSpaceSetCollisionBias(space, cpfpow(1.0f - 0.1f, 60.0f));
    cpSpaceSetCollisionBias(space, cpfpow(1.0f - 0.1f, 1200000.0f));

    terrain = cpSegmentShapeNew(cpSpaceGetStaticBody(space),
        cpv(-TERRAIN_HALF_WIDTH, TERRAIN_Y), cpv(TERRAIN_HALF_WIDTH, TERRAIN_Y), TERRAIN_RADIUS);
    cpShapeSetFriction(terrain, 0.8);
    cpShapeSetElasticity(terrain, 0.5);
    cpShapeSetFilter(terrain, cpShapeFilterNew(0, CATEGORY_TERRAIN, CATEGORY_ENTITY));
    cpSpaceAddShape(space, terrain);
}

ChipmunkCreature::~ChipmunkCreature() {
    for (auto& constraint : constraints) {
        cpSpaceRemoveConstraint(space, constraint.second);
        cpConstraintFree(constraint.second);
    }
    for (auto& shape : shapes) {
        cpSpaceRemoveShape(space, shape.second);
        cpShapeFree(shape.second);
    }
    for (auto& body : bodies) {
        cpSpaceRemoveBody(space, body.second);
        cpBodyFree(body.second);
    }
    cpSpaceRemoveShape(space, terrain);
    cpShapeFree(terrain);
    cpSpaceFree(space);
}

struct ChipmunkEngine::PhysicsWorker {
    std::span<cpSpace* const> spaces;
    cpFloat dt;
    std::binary_semaphore workReady{0};
    std::binary_semaphore workDone{0};
    std::jthread thread{[this](std::stop_token stop) {
        for (;;) {
            workReady.acquire();
            if (stop.stop_requested())
                return;
            stepSpaces(spaces, dt);
            workDone.release();
        }
    }};

    ~PhysicsWorker() {
        thread.request_stop();
        workReady.release();
    }

    void startStep(std::span<cpSpace* const> batch, cpFloat subDt) {
        spaces = batch;
        dt = subDt;
        workReady.release();
    }
};

ChipmunkCreature* ChipmunkEngine::findChipmunkCreatureWithId(unsigned int creatureId) const {
    auto it = chipmunkCreatures.find(creatureId);
    return it == chipmunkCreatures.end() ? nullptr : it->second.get();
}

ChipmunkEngine::ChipmunkEngine()
    : maxThreads(std::clamp(std::thread::hardware_concurrency(), 1u, MAX_PHYSICS_THREADS)) {}

ChipmunkEngine::~ChipmunkEngine() { shutdown(); }

void ChipmunkEngine::initialize() {
    std::lock_guard<std::mutex> lock(data_mutex);
    initialized = true;
}

void ChipmunkEngine::update(float dt) {
    std::lock_guard<std::mutex> lock(data_mutex);
    if (chipmunkCreatures.empty())
        return;

    // Entity shapes collide only with terrain, so each space can be stepped independently.
    spacesToStep.clear();
    spacesToStep.reserve(chipmunkCreatures.size());
    for (const auto& [id, creature] : chipmunkCreatures) {
        spacesToStep.push_back(creature->space);
    }

    const std::span<cpSpace* const> spaces = spacesToStep;
    const size_t threadCount = std::min<size_t>(spaces.size(), maxThreads);
    const size_t workerCount = threadCount - 1;
    const float subDt = dt / SUBSTEPS;
    while (workers.size() < workerCount) {
        workers.push_back(std::make_unique<PhysicsWorker>());
    }

    const size_t creaturesPerThread = spaces.size() / threadCount;
    const size_t extraCreatures = spaces.size() % threadCount;
    size_t begin = creaturesPerThread;
    // Give each worker at most one extra creature and keep the first batch for the simulation thread.
    for (size_t i = 0; i < workerCount; ++i) {
        const size_t count = creaturesPerThread + (i < extraCreatures);
        workers[i]->startStep(spaces.subspan(begin, count), subDt);
        begin += count;
    }
    stepSpaces(spaces.first(creaturesPerThread), subDt);
    for (size_t i = 0; i < workerCount; ++i) {
        workers[i]->workDone.acquire();
    }
}

void ChipmunkEngine::removeCreature(unsigned creature_id) {
    std::lock_guard<std::mutex> lock(data_mutex);
    chipmunkCreatures.erase(creature_id);
}

void ChipmunkEngine::removeAllCreatures() {
    std::lock_guard<std::mutex> lock(data_mutex);
    chipmunkCreatures.clear();
}

void ChipmunkEngine::shutdown() {
    std::lock_guard<std::mutex> lock(data_mutex);
    workers.clear();
    spacesToStep.clear();
    chipmunkCreatures.clear();
    initialized = false;
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
            throw std::runtime_error("[ChipmunkEngine] shape must have at least 1 vertex\n");
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

static void addShapeToChipmunkCreature(ChipmunkCreature& creature, cpBody* body, BodyPart* bodyPart, cpVect bias) {
    cpShape* shape = createShapeForBodyPart(body, bodyPart, bias);
    cpSpaceAddShape(creature.space, shape);
    creature.shapes[bodyPart->getId()] = shape;
    bodyPart->setBodyPosBias(Vector2(bias.x, bias.y));
}

void ChipmunkEngine::addBodyPart(unsigned creature_id, BodyPart* bodyPart) {
    std::lock_guard<std::mutex> lock(data_mutex);
    addBodyPart(*chipmunkCreatures.at(creature_id), bodyPart);
}

void ChipmunkEngine::addBodyPart(ChipmunkCreature& creature, BodyPart* bodyPart)
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
    cpSpaceAddBody(creature.space, cp_body);
    creature.bodies[bodyPart->getId()] = cp_body;

    // Add shape
    addShapeToChipmunkCreature(creature, cp_body, bodyPart, bias);

    // Adding shapes if bodyPart has children
    for (auto& child : bodyPart->getAllChildren()) {
        addShapeToChipmunkCreature(creature, cp_body, child, bias);
    }
}

void ChipmunkEngine::addConstraint(unsigned creature_id, Constraint* constraint) {
    std::lock_guard<std::mutex> lock(data_mutex);
    addConstraint(*chipmunkCreatures.at(creature_id), constraint);
}

void ChipmunkEngine::addConstraint(ChipmunkCreature& creature, Constraint* constraint)
{
    cpBody* bodyA = creature.bodies.at(constraint->getPartA()->getId());
    cpBody* bodyB = creature.bodies.at(constraint->getPartB()->getId());

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
    creature.constraints[constraint->getId()] = cp_constraint;
    cpConstraintSetUserData(cp_constraint, (void*)constraint);
    cpSpaceAddConstraint(creature.space, cp_constraint);
}

void ChipmunkEngine::addCreature(Creature *creature)
{
    std::lock_guard<std::mutex> lock(data_mutex);

    const auto id = creature->getId();
    if (const auto* existing = findChipmunkCreatureWithId(id)) {
        if (existing->creature.get() == creature)
            return;
        throw std::invalid_argument("A creature with this ID is already registered");
    }

    auto physics = std::make_unique<ChipmunkCreature>();
    for (auto& bodyPart : creature->getMainBodyParts()) {
        addBodyPart(*physics, bodyPart);
    }
    for (auto& constraint : creature->getConstraints()) {
        addConstraint(*physics, constraint);
    }
    const auto it = chipmunkCreatures.emplace(id, std::move(physics)).first;
    it->second->creature.reset(creature);
}

// NOTE: Maybe not useful
void ChipmunkEngine::removeBodyPart(unsigned creature_id, BodyPart *bodyPart)
{
    std::lock_guard<std::mutex> lock(data_mutex);
    auto it = chipmunkCreatures.find(creature_id);
    if (it != chipmunkCreatures.end()) {
        ChipmunkCreature* chimpmunkCreature = it->second.get();
        auto bodyIt = chimpmunkCreature->bodies.find(bodyPart->getId());
        if (bodyIt != chimpmunkCreature->bodies.end()) {
            cpBody* body = bodyIt->second;
            cpSpaceRemoveBody(chimpmunkCreature->space, body);
            cpBodyFree(body);
            chimpmunkCreature->bodies.erase(bodyIt);
        }
    }
}

// NOTE: Maybe not useful
void ChipmunkEngine::removeConstraint(unsigned creature_id, Constraint *constraint)
{
    std::lock_guard<std::mutex> lock(data_mutex);
    auto it = chipmunkCreatures.find(creature_id);
    if (it != chipmunkCreatures.end()) {
        ChipmunkCreature* chimpmunkCreature = it->second.get();
        auto constraintIt = chimpmunkCreature->constraints.find(constraint->getId());
        if (constraintIt != chimpmunkCreature->constraints.end()) {
            cpConstraint* constraint = constraintIt->second;
            cpSpaceRemoveConstraint(chimpmunkCreature->space, constraint);
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

    // NOTE: Difference here is always 1 so far, so this may be optimized
    std::map<unsigned, size_t> bodyPartIdToCpBodyId;

    size_t creatureCount = 0;
    for (auto& creature : chipmunkCreatures) {
        if (creatureCount++ >= MAX_PROCESSED_CREATURES) 
            break;
        
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
            bodyPartIdToCpBodyId[obj_body.id] = bodies.size() - 1;
        }
    }

    if (initialized) {
        bodies.emplace_back(0, Vector2(0, 0), 0, 0, Vector2(0, 0));
    }

    creatureCount = 0;
    for (auto& creature : chipmunkCreatures) {
        if (creatureCount++ >= MAX_PROCESSED_CREATURES) 
            break;

        for (auto& shapePair : creature.second->shapes) {
            cpShape* cp_shape = shapePair.second;
            BodyPart* bodyPart = static_cast<BodyPart*>(cpShapeGetUserData(cp_shape));
            if (cp_shape == nullptr || bodyPart == nullptr)
                continue;

            ShapeObject obj_shape;
            obj_shape.id = shapePair.first;
            obj_shape.radius = bodyPart->getRadius();
            obj_shape.vertices = bodyPart->getBiasedVertices();

            auto it = bodyPartIdToCpBodyId.find(bodyPart->getRootParent()->getId());
            if (it != bodyPartIdToCpBodyId.end()) {
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
                auto itA = bodyPartIdToCpBodyId.find(bodyPartA->getId());
                if (itA != bodyPartIdToCpBodyId.end()) {
                    obj_constraint.partA = &bodies.at(itA->second);
                } else {
                    obj_constraint.partA = nullptr;
                    std::cerr << "BodyObject not found for BodyPart ID: " << bodyPartA->getId() << std::endl;
                }

                // partB
                BodyPart* bodyPartB = static_cast<BodyPart*>(cpBodyGetUserData(cpConstraintGetBodyB(cp_constraint)));
                auto itB = bodyPartIdToCpBodyId.find(bodyPartB->getId());
                if (itB != bodyPartIdToCpBodyId.end()) {
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
    
    if (initialized) {
        ShapeObject terrain_shape;
        terrain_shape.body = &bodies.back();
        terrain_shape.id = 0;
        terrain_shape.radius=20;
        terrain_shape.isWorldObj = true;
        constexpr auto extent = TERRAIN_HALF_WIDTH + TERRAIN_RADIUS;
        terrain_shape.vertices = {Vector2(-extent, TERRAIN_Y), Vector2(extent, TERRAIN_Y)};
        
        terrain_shape.initShapeType();
        shapes.push_back(terrain_shape);
    }
}

void ChipmunkEngine::getCreatureAIInputs(std::vector<CreaturePhysicsInputs>& out) {
    std::lock_guard<std::mutex> lock(data_mutex);
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
        data.creature = creature_pair.second->creature.get();

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
    std::lock_guard<std::mutex> lock(data_mutex);
    for (auto& creaturePair : chipmunkCreatures) {
        out.emplace_back(creaturePair.second->creature.get());
    }
}

void ChipmunkEngine::applyAIResults(const std::vector<CreaturePhysicsInputs> &data)
{
    std::lock_guard<std::mutex> lock(data_mutex);
    constexpr float MUSCLE_REST_SCALE = 6.0f;
    // Charge the same for a given fractional change in rest length across muscles.
    constexpr float MUSCLE_WORK_FITNESS_IMPACT = 0.01f;
    constexpr float X_DISTANCE_FITNESS_IMPACT = 0.5f;

    for (const auto& result : data) {
        ChipmunkCreature* creature = findChipmunkCreatureWithId(result.creature->getId());
        if (creature == nullptr)
            continue;
        
        auto muscles = creature->creature->getMuscles();
        if (result.outputs.size() != muscles.size()) {
            std::cout << "ChipmunkEngine::applyAIResults: Got incompatible output size to muscle amount (got "<<result.outputs.size()<<", expected "<<muscles.size()<<")\n";
            continue;
        }
        
        for (size_t i = 0; i < muscles.size(); i++) {
            Constraint* muscle = muscles[i];
            cpConstraint* chipmunkMuscle = creature->constraints[muscle->getId()];

            const float neutral_size = muscle->getNeutralSize();
            const float rest_range = neutral_size * MUSCLE_REST_SCALE;
            const float old_rest = cpDampedSpringGetRestLength(chipmunkMuscle);
            const float new_rest = muscle->clampRest(result.outputs[i] * neutral_size) * MUSCLE_REST_SCALE;
            cpDampedSpringSetRestLength(chipmunkMuscle, new_rest);

            const float output_change = rest_range > 0 ? std::abs(old_rest - new_rest) / rest_range : 0.0f;
            creature->creature->increaseFitness(-output_change * MUSCLE_WORK_FITNESS_IMPACT);
        }
        
        if (!creature->bodies.empty()) {
            const auto cp_pos = cpBodyGetPosition(creature->bodies.begin()->second);
            const Vector2 pos(cp_pos.x, cp_pos.y);
            if (creature->posInitialized) {
                const float distance = pos.x - creature->lastPos.x;
                creature->creature->increaseFitness(distance * X_DISTANCE_FITNESS_IMPACT);
            }
            creature->posInitialized = true;
            creature->lastPos = pos;
        }
    }
}
