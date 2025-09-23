#ifndef CHIPMUNK_ENGINE_H
#define CHIPMUNK_ENGINE_H

#include <mutex>
#include <chipmunk/chipmunk.h>

#include "IPhysicsEngine.h"
#include "BodyPart.h"
#include "Constraint.h"
#include "Vector2.h"
#include "ChipmunkCreature.h"
#include "CreaturePhysicsInputs.h"

class ChipmunkEngine : public IPhysicsEngine {
private:
    cpSpace* space;
    std::map<unsigned, ChipmunkCreature*> chipmunkCreatures;
    
    // A mutex to prevent adding bodies and shapes while computing step
    std::mutex step_mutex;
    // A mutex to protect std::map<unsigned, ChipmunkCreature*> chipmunkCreatures
    std::mutex data_mutex;
    
    std::vector<cpShape*> world_shapes;
    std::vector<cpBody*> world_bodies;

    ChipmunkCreature* findChipmunkCreatureForCreature(Creature* creature);

public:
    ChipmunkEngine();
    ~ChipmunkEngine() override;

    void initialize() override;
    void update(float dt) override;
    void shutdown() override;

    void addShapeToChipmunkCreature(cpBody *body, BodyPart *bodyPart, unsigned creature_id, cpVect bias);

    void addBodyPart(unsigned creature_id, BodyPart *bodyPart) override;
    void addConstraint(unsigned creature_id, Constraint* constraint) override;
    void addCreature(Creature* creature) override;
    void removeBodyPart(unsigned creature_id, BodyPart* bodyPart) override;
    void removeConstraint(unsigned creature_id, Constraint* constraint) override;
    void removeCreature(unsigned creature_id) override;
    void removeAllCreatures() override;

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& constraints
    ) override;
    void getCreatureAIInputs(std::vector<CreaturePhysicsInputs>& out) override;
    void getCreatures(std::vector<Creature *>& out) override;

    void applyAIResults(const std::vector<CreaturePhysicsInputs>& data) override;
};

#endif
