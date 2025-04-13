#ifndef CHIPMUNK_ENGINE_H
#define CHIPMUNK_ENGINE_H

#include <chipmunk/chipmunk.h>

#include "IPhysicsEngine.h"
#include "BodyPart.h"
#include "Constraint.h"
#include "Vector2.h"
#include "Creature.h"

class ChipmunkEngine : public IPhysicsEngine {
public:
    ChipmunkEngine();
    ~ChipmunkEngine() override;

    void initialize() override;
    void update(float dt) override;
    void shutdown() override;
    
    void addBodyPart(BodyPart* bodyPart) override;
    void addConstraint(Constraint* joint) override;
    void addCreature(Creature* creature) override;
    void removeBodyPart(BodyPart* bodyPart) override;
    void removeConstraint(Constraint* joint) override;
    void removeCreature(Creature* creature) override;

    void getRenderObjects(
        std::vector<BodyObject>& bodies,
        std::vector<ShapeObject>& shapes,
        std::vector<ConstraintObject>& joints) const override;


private:
    cpSpace* space;
    std::vector<cpBody*> bodies;
    std::vector<cpShape*> shapes;
    std::vector<cpConstraint*> constraints;
    std::vector<Creature*> creatures;
};

#endif
