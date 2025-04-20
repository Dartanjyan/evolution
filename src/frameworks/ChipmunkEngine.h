#ifndef CHIPMUNK_ENGINE_H
#define CHIPMUNK_ENGINE_H

#include <chipmunk/chipmunk.h>
#include <mutex>

#include "IPhysicsEngine.h"
#include "BodyPart.h"
#include "Constraint.h"
#include "Vector2.h"
#include "Creature.h"

struct ChimpmunkCreature {
    Creature* creature;
    std::map<unsigned, cpBody*> bodies;
    std::map<unsigned, cpShape*> shapes;
    std::map<unsigned, cpConstraint*> constraints;
};

class ChipmunkEngine : public IPhysicsEngine {
public:
    ChipmunkEngine();
    ~ChipmunkEngine() override;

    void initialize() override;
    void update(float dt) override;
    void shutdown() override;
    
    void addBodyPart(unsigned creature_id, BodyPart* bodyPart) override;
    void addConstraint(unsigned creature_id, Constraint* constraint) override;
    void addCreature(Creature* creature) override;
    void removeBodyPart(unsigned creature_id, BodyPart* bodyPart) override;
    void removeConstraint(unsigned creature_id, Constraint* constraint) override;
    void removeCreature(unsigned creature_id) override;

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints) override;


private:
    cpSpace* space;
    std::map<unsigned, ChimpmunkCreature*> creatures;
    
    std::mutex step_mutex;
    
    std::vector<cpShape*> world_shapes;
    std::vector<cpBody*> world_bodies;
};

#endif
