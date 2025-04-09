// Creature.h
#ifndef CREATURE_H
#define CREATURE_H

#include <vector>
#include <memory>
#include <map>
#include <functional>
#include "BodyPart.h"
#include "Joint.h"

class Creature {
private:
    unsigned id;
    // only these parts that don't have parent
    std::vector<BodyPart*> bodyParts;
    std::vector<Joint*> joints;
    // Brain* brain;
    float fitness;
    
    unsigned immunity=0;
    
    static unsigned last_id;

public:
    Creature();
    Creature(std::vector<BodyPart*> bodyParts, 
        std::vector<Joint*> joints, 
        unsigned immunity = 0);
    Creature(const Creature& other);
    ~Creature();
    
    unsigned getId() const { return id; }
    const std::vector<Joint*>& getJoints() const { return joints; }
    float getFitness() const { return fitness; }
    
    void setFitness(float value) { fitness = value; }
    
    void addJoint(Joint* joint);
    void removeJoint(Joint* joint);
    
    // Return all the body parts recursevely
    std::vector<BodyPart*> getAllBodyParts() const;
    
    static Creature* createBasicCreature();
    
    static unsigned newId();
    static void resetId();
};

#endif
