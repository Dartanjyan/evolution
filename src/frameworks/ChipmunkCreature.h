#ifndef CHIPMUNK_CREATURE_H
#define CHIPMUNK_CREATURE_H

#include <map>
#include <chipmunk/chipmunk.h>
#include "Creature.h"

struct ChipmunkCreature {
    Creature* creature;
    std::map<unsigned, cpBody*> bodies;
    std::map<unsigned, cpShape*> shapes;
    std::map<unsigned, cpConstraint*> constraints;
    Vector2 lastPos;
    bool posInitialized = false;
};

#endif
