#include "Creature.h"
//#include <queue>
#include <algorithm>

unsigned Creature::last_id = 0;

Creature::Creature() 
    : id(Creature::newId()),
      bodyParts(std::vector<BodyPart*> {}),
      constraints(std::vector<Constraint*> {}),
      brain(nullptr),
      fitness(0.0f),
      immunity(0)
{
    // std::cout << "Creating empty Creature, id = " << id << "\n";
}

Creature::Creature(std::vector<BodyPart*> bodyParts,
                std::vector<Constraint*> constraints,
                Brain* brain,
                unsigned immunity): 
    id(Creature::newId()), 
    bodyParts(bodyParts),
    constraints(constraints),
    brain(brain),
    fitness(0.0f),
    immunity(immunity)
{
    // std::cout << "Creating Creature, id = " << id << "\n";
}

Creature::Creature(const Creature &other):
    id(Creature::newId()),
    fitness(other.fitness),
    immunity(other.immunity)
{
    // std::cout << "Copying Creature, id "<<other.id<<"->"<<id<< "\n";

    std::map<BodyPart*, BodyPart*> partMapping;

    for (BodyPart* oldPart : other.bodyParts) {
        BodyPart* newPart = new BodyPart(*oldPart, nullptr);
        bodyParts.push_back(newPart);

        partMapping[oldPart] = newPart;

        std::function<void(BodyPart*, BodyPart*)> mapBodyPartsRecursively =
            [&partMapping, &mapBodyPartsRecursively](BodyPart* oldPart, BodyPart* newPart) {
                partMapping[oldPart] = newPart;

                if (oldPart->getChildren().size() != newPart->getChildren().size()) {
                    return;
                }

                for (size_t i = 0; i < oldPart->getChildren().size(); ++i) {
                    mapBodyPartsRecursively(oldPart->getChildren()[i], newPart->getChildren()[i]);
                }
            };

        mapBodyPartsRecursively(oldPart, newPart);
    }

    for (Constraint* oldConstraint : other.constraints) {
        BodyPart* oldBodyA = oldConstraint->getPartA();
        BodyPart* oldBodyB = oldConstraint->getPartB();

        auto itA = partMapping.find(oldBodyA);
        auto itB = partMapping.find(oldBodyB);

        if (itA != partMapping.end() && itB != partMapping.end()) {
            BodyPart* newBodyA = itA->second;
            BodyPart* newBodyB = itB->second;

            Constraint* newConstraint = new Constraint(
                newBodyA, newBodyB, oldConstraint->getType(),
                oldConstraint->getAnchorA(), oldConstraint->getAnchorB(),
                oldConstraint->getRest(), oldConstraint->getStiffness(),
                oldConstraint->getDamping(), oldConstraint->getCollideConnected()
            );
            constraints.push_back(newConstraint);
        }
    }
    
    brain = new Brain(*other.brain);
}

Creature::~Creature()
{
    // std::cout << "Deleting Creature, id = "<<id<<"\n";
    for (Constraint* constraint : constraints) {
        delete constraint;
    }
    constraints.clear();

    for (BodyPart* part : bodyParts) {
        delete part;
    }
    bodyParts.clear();

    delete brain;
}

void Creature::replaceBrain(Brain *new_brain)
{
    delete brain;
    brain = new_brain;
}

void Creature::addConstraint(Constraint *constraint)
{
    if (constraint) {
        constraints.push_back(constraint);
    }
}

void Creature::removeConstraint(Constraint* constraint)
{
    if (!constraint) return;
    
    auto it = std::find(constraints.begin(), constraints.end(), constraint);
    if (it != constraints.end()) {
        delete *it;
        constraints.erase(it);
    }
}

std::vector<BodyPart*> Creature::getAllBodyParts() const
{
    std::vector<BodyPart*> allParts;
    
    std::function<void(BodyPart*)> collectParts = [&allParts, &collectParts](BodyPart* part) {
        allParts.push_back(part);
        
        for (BodyPart* child : part->getChildren()) {
            collectParts(child);
        }
    };
    
    for (BodyPart* rootPart : bodyParts) {
        collectParts(rootPart);
    }
    
    return std::vector<BodyPart*>(allParts);
}

const BodyPart* Creature::getBodyPartById(unsigned id) const
{
    for (auto* bodyPart : this->getAllBodyParts()) {
        if (bodyPart->getId() == id) { 
            return bodyPart;
        }
    }
    return nullptr;
}

Creature* Creature::createBasicCreature()
{
    /*
    p1 = myCreature.PolySegment(
        self.space, 1, None,
        (
            pymunk.Vec2d(40, 20),
            pymunk.Vec2d(140, 20),
            pymunk.Vec2d(120, 50)
        )
    )

    ll1 = myCreature.Bone(self.space, 1, pymunk.Vec2d(40, 20), pymunk.Vec2d(80, 40)) 
    ll2 = myCreature.Bone(self.space, 2, pymunk.Vec2d(80, 40), pymunk.Vec2d(40, 60))
    ll3 = myCreature.Bone(self.space, 3, pymunk.Vec2d(40, 60), pymunk.Vec2d(60, 80))
    rl1 = myCreature.Bone(self.space, 4, pymunk.Vec2d(140, 20), pymunk.Vec2d(160, 40))
    rl2 = myCreature.Bone(self.space, 5, pymunk.Vec2d(160, 40), pymunk.Vec2d(140, 60))
    rl3 = myCreature.Bone(self.space, 6, pymunk.Vec2d(140, 60), pymunk.Vec2d(160, 60))
    t1 = myCreature.Bone(self.space, 7, pymunk.Vec2d(40, 20), pymunk.Vec2d(0, 0))
    h1 = myCreature.Bone(self.space, 8, pymunk.Vec2d(140, 20), pymunk.Vec2d(160, 0))

    stiffness = 8e5
    damping = 4e4
    j1 = myCreature.Joint(self.space, 1, p1.body, ll1.body, pymunk.Vec2d(40, 20), pymunk.Vec2d(40, 20), stiffness=stiffness, damping=damping)
    j2 = myCreature.Joint(self.space, 2, ll1.body, ll2.body, pymunk.Vec2d(80, 40), pymunk.Vec2d(80, 40), stiffness=stiffness, damping=damping)
    j3 = myCreature.Joint(self.space, 3, ll2.body, ll3.body, pymunk.Vec2d(40, 60), pymunk.Vec2d(40, 60), stiffness=stiffness, damping=damping)
    j4 = myCreature.Joint(self.space, 4, p1.body, rl1.body, pymunk.Vec2d(140, 20), pymunk.Vec2d(140, 20), stiffness=stiffness, damping=damping)
    j5 = myCreature.Joint(self.space, 5, rl1.body, rl2.body, pymunk.Vec2d(160, 40), pymunk.Vec2d(160, 40), stiffness=stiffness, damping=damping)
    j6 = myCreature.Joint(self.space, 6, rl2.body, rl3.body, pymunk.Vec2d(140, 60), pymunk.Vec2d(140, 60), stiffness=stiffness, damping=damping)
    j7 = myCreature.Joint(self.space, 7, p1.body, t1.body, pymunk.Vec2d(40, 20), pymunk.Vec2d(40, 20), stiffness=stiffness, damping=damping)
    j8 = myCreature.Joint(self.space, 8, p1.body, h1.body, pymunk.Vec2d(140, 20), pymunk.Vec2d(140, 20), stiffness=stiffness, damping=damping)

    polies.extend([p1])
    bones.extend([ll1, ll2, ll3, rl1, rl2, rl3, t1, h1])
    joints.extend([j1, j2, j3, j4, j5, j6, j7, j8])
    */
    float scale = 1;
    Vector2 bias(100, 000);

    BodyPart* body = new BodyPart(nullptr, { Vector2(40, 20)*scale+bias, Vector2(140, 20)*scale+bias, Vector2(120, 50)*scale+bias}, false, 0);
    
    float radius = 6;
    BodyPart* ll1 = new BodyPart(nullptr, {Vector2(40, 20)*scale+bias, Vector2(80, 40)*scale+bias}, false, radius);
    BodyPart* ll2 = new BodyPart(nullptr, {Vector2(80, 40)*scale+bias, Vector2(40, 60)*scale+bias}, false, radius);
    BodyPart* ll3 = new BodyPart(nullptr, {Vector2(40, 60)*scale+bias, Vector2(60, 80)*scale+bias}, false, radius);
    BodyPart* rl1 = new BodyPart(nullptr, {Vector2(140, 20)*scale+bias, Vector2(160, 40)*scale+bias}, false, radius);
    BodyPart* rl2 = new BodyPart(nullptr, {Vector2(160, 40)*scale+bias, Vector2(140, 60)*scale+bias}, false, radius);
    BodyPart* rl3 = new BodyPart(nullptr, {Vector2(140, 60)*scale+bias, Vector2(160, 60)*scale+bias}, false, radius);
    BodyPart* t1 = new BodyPart(nullptr, {Vector2(40, 20)*scale+bias, Vector2(0, 0)*scale+bias}, false, radius);
    BodyPart* h1 = new BodyPart(nullptr, {Vector2(140, 20)*scale+bias, Vector2(160, 0)*scale+bias}, false, radius);
    
    Constraint* j1 = new Constraint(body, ll1, ConstraintType::JOINT, Vector2(40, 20)*scale+bias, false);
    Constraint* j2 = new Constraint(ll1, ll2, ConstraintType::JOINT, Vector2(80, 40)*scale+bias, false);
    Constraint* j3 = new Constraint(ll2, ll3, ConstraintType::JOINT, Vector2(40, 60)*scale+bias, false);
    Constraint* j4 = new Constraint(body, rl1, ConstraintType::JOINT, Vector2(140, 20)*scale+bias, false);
    Constraint* j5 = new Constraint(rl1, rl2, ConstraintType::JOINT, Vector2(160, 40)*scale+bias, false);
    Constraint* j6 = new Constraint(rl2, rl3, ConstraintType::JOINT, Vector2(140, 60)*scale+bias, false);
    Constraint* j7 = new Constraint(body, t1, ConstraintType::JOINT, Vector2(40, 20)*scale+bias, false);
    Constraint* j8 = new Constraint(body, h1, ConstraintType::JOINT, Vector2(140, 20)*scale+bias, false);
    
    float stiffness = 4e2;
    float damping = 4e1;
    float rest = 0;
    Constraint* m1 = new Constraint(
        body, ll1,
        ConstraintType::MUSCLE, 
        Vector2(90, 20)*scale+bias, 
        Vector2(60, 30)*scale+bias, 
        rest, stiffness*2, damping);
    Constraint* m2 = new Constraint(
        ll1, ll2,
        ConstraintType::MUSCLE, 
        Vector2(60, 30)*scale+bias, 
        Vector2(60, 50)*scale+bias, 
        rest, stiffness, damping);
    Constraint* m3 = new Constraint(
        ll2, ll3,
        ConstraintType::MUSCLE, 
        Vector2(60, 50)*scale+bias, 
        Vector2(50, 70)*scale+bias, 
        rest, stiffness, damping);
    Constraint* m4 = new Constraint(
        body, rl1,
        ConstraintType::MUSCLE, 
        Vector2(130, 35)*scale+bias, 
        Vector2(150, 30)*scale+bias, 
        rest, stiffness, damping);
    Constraint* m5 = new Constraint(
        rl1, rl2,
        ConstraintType::MUSCLE, 
        Vector2(150, 30)*scale+bias, 
        Vector2(150, 50)*scale+bias, 
        rest, stiffness, damping/2);
    Constraint* m6 = new Constraint(
        rl2, rl3,
        ConstraintType::MUSCLE, 
        Vector2(150, 50)*scale+bias, 
        Vector2(150, 60)*scale+bias, 
        rest, stiffness, damping);
    Constraint* m7 = new Constraint(
        body, t1,
        ConstraintType::MUSCLE, 
        Vector2(90, 20)*scale+bias, 
        Vector2(20, 10)*scale+bias, 
        rest, stiffness, damping);
    Constraint* m8 = new Constraint(
        body, h1,
        ConstraintType::MUSCLE, 
        Vector2(90, 20)*scale+bias, 
        Vector2(150, 10)*scale+bias, 
        rest, stiffness, damping);
    Constraint* m9 = new Constraint(
        body, ll1,
        ConstraintType::MUSCLE, 
        Vector2(90, 40)*scale+bias, 
        Vector2(60, 30)*scale+bias, 
        rest, stiffness, damping);
    
    std::vector<BodyPart*> bodyParts = {body, ll1, ll2, ll3, rl1, rl2, rl3, t1, h1};
    std::vector<Constraint*> constraints = {j1, j2, j3, j4, j5, j6, j7, j8, m1, m2, m3, m4, m5, m6, m7, m8};
    
    std::vector<size_t> layers = {1};
    Brain* brain = new Brain({2, 3, 4});

    Creature* creature = new Creature(bodyParts, constraints, brain);
    
    return creature;
}

unsigned Creature::newId() { return ++Creature::last_id; }

void Creature::resetId() { Creature::last_id = 0; }
