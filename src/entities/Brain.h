#ifndef BRAIN_H
#define BRAIN_H

#include <vector>

class Brain {
	// TODO: Do something with layer_sizes. Maybe remove it from constructor args
public:
    Brain();
    Brain(
        const std::vector<size_t>& layer_sizes, 
        const std::vector<double>& weights = {}, 
        const std::vector<double>& biases = {}
    );

    unsigned getId() { return id; }
    const std::vector<size_t>& getLayerSizes() const noexcept;
    const std::vector<double>& getWeights() const noexcept;
    const std::vector<double>& getBiases() const noexcept;

	// Genome is simply a single vector like (weights_ + biases_)
    
	std::vector<double> encodeGenome() const;
    void decodeGenome(const std::vector<double>& genome);
    
	void setWeights(const std::vector<double>& w);
    void setBiases (const std::vector<double>& b);

	static unsigned newId();
    static void resetId();
private:
	unsigned id;
	static unsigned last_id;

    std::vector<size_t> layer_sizes_;

    // Flat array: [i][j] will be [i*column + j]
    std::vector<double> weights_;
    std::vector<double> biases_;
};

#endif // BRAIN_H
