#include <fann.h>
#include <iostream>
#include <vector>
#include <omp.h>

#define INPUT 5
#define OUTPUT 3
#define MAX_LAYER_SIZE 10
#define NB_HIDDEN_LAYER 2;

struct fann* create_network(const std::vector<unsigned int>& prop) {
	if (prop.size() < 2) {
		std::cerr << "ERROR :: NETWORK MUST CONTAINED AT LEAST 2 LAYERS" << std::endl;
		return nullptr;
	}
	
	struct fann* temp = fann_create_standard_array(prop.size(),prop.data());

	fann_set_activation_function_hidden(temp, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(temp, FANN_SIGMOID_SYMMETRIC);

	return temp;
}

int main(int argc, char* argv[]) {
	// on fixe le nombre de threads dans le programme
	omp_set_num_threads(omp_get_max_threads());


	struct fann* network;
	struct fann_train_data* train_data, * test_data;
	fann_type* output_value;
	enum fann_train_enum training_algorithm = FANN_TRAIN_RPROP;
	enum fann_activationfunc_enum activation;
	fann_type steepness;
	unsigned int bit_fail_train, bit_fail_test;
	float mse_train, mse_test;
	int multi = 0;
	//
	// Récupération des données 
	//
	train_data = fann_read_train_from_file();
	test_data = fann_read_train_from_file();

	fann_scale_train_data(train_data, -1, 1);
	fann_scale_train_data(test_data, -1, 1);

	//
	// Creation du réseau de neurones
	//

	std::vector<unsigned int> network_prop = {2,10,1 };
	struct fann* network = create_network(network_prop);

	if (network == nullptr) {
		std::cout << "ERROR :: NEURAL NETWORK CREATION FAILED " << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "Neural Network creation SUCCEED " << std::endl;

	//
	// Network properties 
	//
	fann_set_training_algorithm(network, training_algorithm);
	fann_set_activation_function_hidden(network, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(network, FANN_LINEAR);
	fann_set_train_error_function(network, FANN_ERRORFUNC_LINEAR);
	
	if (!multi) {
		steepness = 1;
		fann_set_cascade_activation_steepnesses(network, &steepness, 1);
		activation = FANN_SIGMOID_SYMMETRIC;

		fann_set_cascade_activation_functions(network, &activation, 1);
		fann_set_cascade_num_candidate_groups(network, 8);
	}


	std::cout << "Training properties set correctly " << std::endl;


	// Test après entrainements

	fann_type *input_value = new fann_type[INPUT];
	output_value = new fann_type[OUTPUT];

	#pragma omp parallel for
	for (int i = 0; i < INPUT; ++i) {
		input_value[i] = (float)rand() / (float)RAND_MAX;
	}

	std::cout << "Input value set with random values " << std::endl;

	// Calcul dans le réseau
	fann_type *shared_input = new fann_type[MAX_LAYER_SIZE];
	fann_type *shared_output = new fann_type[MAX_LAYER_SIZE];

	memcpy(shared_input, input_value, sizeof(fann_type) * INPUT);
	unsigned int layer_size = 2 + NB_HIDDEN_LAYER;

	// Récupération de la matrice des poids
	
	unsigned int total_connection = fann_get_total_connections(network);

	if (total_connection == 0) {
		std::cerr << "ERROR :: This network do not have any connections " << std::endl;
		fann_destroy(network);
		return EXIT_FAILURE;
	}

	std::cout << "Total connection : " << total_connection << std::endl;

	fann_type* weigth_array = new fann_type[total_connection];
	fann_get_weights(network, weigth_array);
	for (int i = 0; i < total_connection; ++i) {
		std::cout << weigth_array[i] << std::endl;
	}
	std::cout << "Weigth matrix loaded " << std::endl;

	unsigned int num_weigths;
	unsigned int first_weigth_index = 0;

	std::cout << "Start neural network calculation " << std::endl;

	for (unsigned int i=0; i < layer_size-1; ++i){

		std::cout << "couche " << i << " et " << i + 1 << " START" << std::endl;
		
		int input_size = network_prop[i];
		int output_size = network_prop[i + 1];
		
		num_weigths = (input_size+1) * output_size;
		std::cout << first_weigth_index << " " << num_weigths << std::endl;
		// Calculate value of each next neurons in one layer
		#pragma omp parallel for
		for (int j = 0; j < output_size; ++j) {
			float sum = 0;
			for (int k = 0; k < input_size; ++k) {
				sum += shared_input[k] * weigth_array[first_weigth_index + k*output_size + j];
			}
			shared_output[j] = 1.0f / (1.0f * expf(-sum)); // exemple de fonctions sigmoidale
		}

		memcpy(shared_input, shared_output, sizeof(fann_type) * output_size);
		first_weigth_index += num_weigths;
		
		std::cout << "couche " << i << " END" << std::endl;
	}

	memcpy(output_value, shared_input, sizeof(fann_type) * OUTPUT);

	std::cout << "Neural network calculation DONE " << std::endl;
	std::cout << output_value[0];


	delete[] shared_input;
	delete[] shared_output;
	fann_destroy(network);

	return EXIT_SUCCESS;
}