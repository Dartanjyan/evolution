#ifndef CHIPMUNK_ENGINE_H
#define CHIPMUNK_ENGINE_H

#include <chipmunk/chipmunk.h>

#include "IPhysicsEngine.h"

class ChipmunkEngine : public IPhysicsEngine {
public:
    ChipmunkEngine();
    ~ChipmunkEngine() override;

    void initialize() override;
    void update(float dt) override;
    void shutdown() override;

    // cpSpace* getSpace() const { return space; }

private:
    cpSpace* space;
};

#endif
