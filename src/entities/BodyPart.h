#ifndef BODYPART_H
#define BODYPART_H

class BodyPart {
public:
    BodyPart(unsigned int id, 
        float density, 
        float friction, 
        float mass, 
        float elasticity, 
        bool isRootPart, 
        bool isSensorPart);
    BodyPart(const BodyPart&);
    ~BodyPart();

    // getters
    unsigned int getId() const { return id; }
    float getDensity() const { return density; }
    float getFriction() const { return friction; }
    float getMass() const { return mass; }
    float getElasticity() const { return elasticity; }
    bool isRoot() const { return isRootPart; }
    bool isSensor() const { return isSensorPart; }
    // setters
    void setId(unsigned int id) { this->id = id; }
    void setDensity(float density) { this->density = density; }
    void setFriction(float friction) { this->friction = friction; }
    void setMass(float mass) { this->mass = mass; }
    void setElasticity(float elasticity) { this->elasticity = elasticity; }
    void setRoot(bool isRoot) { this->isRootPart = isRoot; }
    void setSensor(bool isSensor) { this->isSensorPart = isSensor; }

    // setters for vertices and connected_to
    // void setVertices(const std::vector<Vertex>& vertices) { this->vertices = vertices; }
    // void setConnectedTo(const std::vector<unsigned int>& connected_to) { this->connected_to = connected_to; }
    // getters for vertices and connected_to
    // std::vector<Vertex> getVertices() const { return vertices; }
    // std::vector<unsigned int> getConnectedTo() const { return connected_to; }

private:
    unsigned int id;
    float density;
    float friction;
    float mass;
    float elasticity;
    bool isRootPart;
    bool isSensorPart;

    // vertices
    // connected_to
};

#endif