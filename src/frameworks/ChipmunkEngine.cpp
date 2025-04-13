#include "ChipmunkEngine.h"

ChipmunkEngine::ChipmunkEngine() : space(nullptr) {}

ChipmunkEngine::~ChipmunkEngine() {
    shutdown();
}

void ChipmunkEngine::initialize() {
    space = cpSpaceNew();
    cpSpaceSetGravity(space, cpv(0, -100));
    // Here I may add more settings
}

void ChipmunkEngine::update(float dt) {
    cpSpaceStep(space, dt);
}

void ChipmunkEngine::shutdown() {
    if (space) {
        cpSpaceFree(space);
        space = nullptr;
    }
}
