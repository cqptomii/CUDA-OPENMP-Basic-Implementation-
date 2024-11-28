
#include <doublefann.h>
#include <fann_cpp.h>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
	const std::vector<unsigned int> network_prop = { 200,500,500,500,500,500,500,500,500,500,500,100 };
	const float learning_rate = 0.7f;
	const float desired_error = 0.0001f;
	const unsigned int max_iterations = 300000;
	const unsigned int iterations_between_reports = 1000;

	//
	//	Network creation
	//

	FANN::neural_net network;
	if (!network.create_standard_array(network_prop.size(), network_prop.data())) {
		std::cerr << "ERROR :: UNABLE TO CREATE NEURAL NETWORK " << std::endl;
		return EXIT_FAILURE;
	}

	//
	// Network properties settings
	//

	network.set_learning_rate(learning_rate);
	network.set_activation_steepness_hidden(1.0);
	network.set_activation_steepness_output(1.0);

	network.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	network.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

	network.set_training_algorithm(FANN::TRAIN_RPROP);
	network.randomize_weights(-.1f, .1f);

	// Print network parameter 
	//
	std::cout << std::endl;
	switch (network.get_network_type()) {
	case FANN::LAYER:
		std::cout << "LAYER" << std::endl;
		break;
	case FANN::SHORTCUT:
		std::cout << "SHORTCUT" << std::endl;
		break;
	default:
		std::cout << "OTHER" << std::endl;
		break;
	}
	network.print_parameters();

	//
	// Training on data  ( Plus tard )
	//

	//
	// Test du réseau 
	// 
	
	fann_type* input = new fann_type[200];
	fann_type* output = new fann_type[100];

	std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;


	for (int i = 0; i < 200; ++i) {
		input[i] = (float)rand() / (float)RAND_MAX;
	}

	start = std::chrono::high_resolution_clock::now();
	output = network.run(input);
	stop = std::chrono::high_resolution_clock::now();

	std::cout << "Calculation end" << std::endl;

	for (int i = 0; i < 100; ++i) {
		std::cout << output[i] << std::endl;
	}

	std::chrono::duration<double> time_elapsed = stop - start;

	std::cout << "Calcul du reseau en : " << time_elapsed.count() << std::endl;

	network.destroy();

	delete[] input;
	input = nullptr;

	return EXIT_SUCCESS;
}