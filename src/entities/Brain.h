// Written in vim btw :>

#ifndef BRAIN_H
#define BRAIN_H
#include <vector>
#include <iostream>

class Brain {
private:
	static unsigned last_id;

	unsigned id;
	const std::vector<unsigned short> layers;
	std::vector<double> memory;

	// Maybe create Layer class later but not sure
	std::vector<std::vector<double>> weights;
	std::vector<double> biasWeights;
public:
	Brain(const std::vector<unsigned short> layers, std::vector<std::vector<double>> weights, std::vector<double> biasWeights);
	Brain(const Brain &other);
	~Brain();

	unsigned getId() const { return id; }
	std::vector<unsigned short> getLayers() const { return layers; }
	std::vector<std::vector<double>> getWeights() const { return weights; }

	void setWeights(const std::vector<std::vector<double>> &new_weights) { weights = new_weights; }
	void setMemory(const std::vector<double> &new_memory) { memory = new_memory; }

	static unsigned newId() { return ++last_id; };
    static void resetId() { last_id = 0; };
};

#endif

