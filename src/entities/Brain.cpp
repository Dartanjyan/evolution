#include "Brain.h"

unsigned Brain::last_id = 0;

Brain::Brain(const std::vector<unsigned short> layers, 
	std::vector<std::vector<double>> weights, 
	std::vector<double> biasWeights): 
	id(Brain::newId()),
	layers(layers), 
	weights(weights),
	biasWeights(biasWeights)
{
	// std::cout << "Creating new Brain, id = "<<id<<"\n";
}

Brain::Brain(const Brain &other):
	id(Brain::newId()),
	layers(other.layers),
	memory(other.memory),
	weights(other.weights),
	biasWeights(other.biasWeights)
{
	// std::cout << "Copying Brain, id "<<other.id<<"->"<<id<<"\n";
}

Brain::~Brain()
{
	// std::cout << "Deleting Brain, id = "<<id<<"\n";
}
