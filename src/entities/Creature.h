// Creature.h
#ifndef CREATURE_H
#define CREATURE_H

#include <vector>
#include <memory>
#include "BodyPart.h"
#include "Joint.h"

class Creature {
private:
    unsigned id;
    BodyPart* rootPart;
    std::vector<BodyPart*> bodyParts;
    std::vector<Joint*> joints;
    float fitness;
    
    unsigned immunity=0;
    
    static unsigned last_id;

public:
    Creature();
    Creature(BodyPart* rootPart, 
        std::vector<BodyPart*> bodyParts, 
        std::vector<Joint*> joints, 
        unsigned immunity = 0);
    Creature(const Creature& other);
    ~Creature();
    
    unsigned getId() const { return id; }
    BodyPart* getRootPart() const { return rootPart; }
    const std::vector<Joint*>& getJoints() const { return joints; }
    float getFitness() const { return fitness; }
    
    void setRootPart(BodyPart* part);
    void setFitness(float value) { fitness = value; }
    
    void addJoint(Joint* joint);
    void removeJoint(Joint* joint);
    
    std::vector<BodyPart*> getAllBodyParts() const;
    std::vector<Joint*> getAllJoints() const;
    
    static Creature* createBasicCreature();
    
    static unsigned newId();
    static void resetId();
};

#endif