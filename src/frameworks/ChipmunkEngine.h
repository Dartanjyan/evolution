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
    struct PhysicsWorker;

    bool initialized = false;
    std::map<unsigned, std::unique_ptr<ChipmunkCreature>> chipmunkCreatures;
    std::vector<std::unique_ptr<PhysicsWorker>> workers;
    std::vector<cpSpace*> spacesToStep;
    const unsigned maxThreads;

    // Keep space mutations and render snapshots outside the parallel step.
    std::mutex data_mutex;

    ChipmunkCreature* findChipmunkCreatureWithId(unsigned int creatureId) const;
    void addBodyPart(ChipmunkCreature& creature, BodyPart* bodyPart);
    void addConstraint(ChipmunkCreature& creature, Constraint* constraint);

public:
    ChipmunkEngine();
    ~ChipmunkEngine() override;

    void initialize() override;
    void update(float dt) override;
    void shutdown() override;

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
