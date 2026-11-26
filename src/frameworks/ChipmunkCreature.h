#ifndef CHIPMUNK_CREATURE_H
#define CHIPMUNK_CREATURE_H

#include <map>
#include <memory>
#include <chipmunk/chipmunk.h>
#include "Creature.h"

struct ChipmunkCreature {
    ChipmunkCreature();
    ~ChipmunkCreature();

    std::unique_ptr<Creature> creature;
    cpSpace* space;
    cpShape* terrain;
    std::map<unsigned, cpBody*> bodies;
    std::map<unsigned, cpShape*> shapes;
    std::map<unsigned, cpConstraint*> constraints;
    Vector2 lastPos;
    bool posInitialized = false;
};

#endif
