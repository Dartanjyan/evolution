#ifndef BRAIN_H
#define BRAIN_H

#include <vector>
//#include <cstddef>
#define DEFAULT_MEMORY_SIZE 4

class Brain {
private:
	unsigned id;
	static unsigned last_id;
    
    std::vector<std::size_t> layer_sizes_;

    // Flat array: [i][j] will be [i*column + j]
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
    std::vector<double> memory_;
public:
    Brain();
    Brain(
        const std::vector<std::size_t>& layer_sizes,
        const std::size_t memory = DEFAULT_MEMORY_SIZE,
        const std::vector<std::vector<double>>& weights = {}, 
        const std::vector<double>& biases = {}
    );

    unsigned getId() { return id; }
    const std::vector<std::size_t>& getLayerSizes() const noexcept;
    const std::vector<std::vector<double>>& getWeights() const noexcept;
    const std::vector<double>& getBiases() const noexcept;
    const std::vector<double>& getMemory() const noexcept;

	// Genome is simply a single vector like (weights_ + biases_)
    
	// std::vector<double> encodeGenome() const;
    // void decodeGenome(const std::vector<double>& genome);
    
	void setWeights(const std::vector<std::vector<double>>& new_weights);
    void setBiases (const std::vector<double>& new_biases);
    void setMemory(const std::vector<double>& new_memory);

	static unsigned newId();
    static void resetId();
};

#endif // BRAIN_H
