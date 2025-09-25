// Creature.h
#ifndef CREATURE_H
#define CREATURE_H

#include <vector>
#include "BodyPart.h"
#include "Constraint.h"
#include "Brain.h"

class Creature {
private:
    unsigned id;
    // only these parts that don't have a parent
    std::vector<BodyPart*> bodyParts;
    std::vector<Constraint*> constraints;
    Brain* brain;
    float fitness;
    
    unsigned immunity=0;
    
    static unsigned last_id;

public:
    Creature();
    Creature(std::vector<BodyPart*> bodyParts, 
        std::vector<Constraint*> constraints, 
        Brain* brain,
        unsigned immunity = 0);
    // Creature(const Creature& other);
    ~Creature();
    
    unsigned getId() const { return id; }
    const std::vector<Constraint*> getConstraints() const { return constraints; }
    Brain* getBrain() const { return brain; }
    float getFitness() const { return fitness; }
    
    void setFitness(float value) { fitness = value; }
    void replaceBrain(Brain* new_brain);
    
    void addConstraint(Constraint* constraint);
    void removeConstraint(Constraint* constraint);
    
    // Return all the body parts recursevely
    std::vector<BodyPart*> getAllBodyParts() const;
    // Return all the body parts that are not children of other parts
    std::vector<BodyPart*> getMainBodyParts() const { return bodyParts; }
    const BodyPart* getBodyPartById(unsigned id) const;
    std::vector<Constraint*> getJoints() const;
    std::vector<Constraint*> getMuscles() const;
    std::vector<BodyPart*> getSightParts() const;
    
    
    // Set new memory vector in Brain and apply
    // new muscle restLength (and stiffness?)
    // void applyAIResults(std::vector<float> results);

    static Creature* createBasicCreature(Brain* brain = nullptr);
    
    static unsigned newId();
    static void resetId();
};

#endif
