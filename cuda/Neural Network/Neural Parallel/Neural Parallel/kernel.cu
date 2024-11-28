
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <floatfann.h>
#include <fann_cpp.h>
#include <iostream>
#include <chrono>
#include <vector>

#define INPUT 200
#define OUTPUT 100
#define HIDE_LAYER 10
#define NB_LAYER HIDE_LAYER+2
#define LAYER_MAX_SIZE 500

__global__ void cuda_fann_run(fann_type *input, fann_type* output, fann_type* weigths, int input_size,int output_size){
	__shared__ float s_input[LAYER_MAX_SIZE];
	__shared__ float s_output[LAYER_MAX_SIZE];
	unsigned int  idX = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idX < input_size) {
		s_input[idX] = input[idX];
	}
	
	//Synchronize allocation of shared space
	__syncthreads();

	//Reset the sum
	float sum = 0.0f;
	
	if (idX < output_size) {
		for (int i = 0; i < input_size; ++i) {
			sum += s_input[i] * weigths[i * output_size + idX];
		}
		s_output[idX] = 1.0f / (1.0f + expf(-sum)); // activation sigmoÏde
	}

	//Copy shared output value in global memory
	if (idX < output_size) {
		output[idX] = s_output[idX];
	}
}
fann_type* fann_get_layer_weights(fann* network, unsigned int first_weight_pos, unsigned int amount) {
	fann_type* weight_matrix = new fann_type[amount];

	for (int i = 0; i < amount; ++i) {
		weight_matrix[i] = network->weights[first_weight_pos + i];
	}

	return weight_matrix;
}

int main(){
	std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
	std::chrono::duration<double> time_elapsed;
	std::vector<unsigned int> network_prop = { INPUT,400,500,500,500,500,500,500,500,500,500,OUTPUT };
	const float rate = 0.7f;
	struct fann *network;
	
	//
	//	Création du réseau
	//
	if ((network = fann_create_standard_array(network_prop.size(), network_prop.data())) == nullptr) {
		std::cerr << "ERROR :: NETWORK CREATION FAILED" << std::endl;
		return EXIT_FAILURE;
	}
	//
	// Network settings
	//

	fann_set_learning_rate(network,rate);
	
	fann_set_activation_function_hidden(network,FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(network,FANN_SIGMOID_SYMMETRIC);
	
	fann_set_activation_steepness_hidden(network,1.0);
	fann_set_activation_steepness_output(network,1.0);

	fann_set_training_algorithm(network, FANN_TRAIN_RPROP);
	//
	// Print network parameter 
	//
	std::cout << std::endl;
	switch (fann_get_network_type(network)) {
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
	fann_print_parameters(network);

	//
	// Teste du réseau sur un jeu de données 
	//

	fann_type* h_input = new fann_type[INPUT];
	fann_type *h_output = new fann_type[OUTPUT];
	fann_type* h_weight;
	for (int i = 0; i < INPUT; ++i) {
		h_input[i] = (float)rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < fann_get_total_connections(network); ++i) {
		network->weights[i] = (float)rand() / (float)RAND_MAX;
	}
	//
	//	PARTIE SEQUENTIEL
	//
	start = std::chrono::high_resolution_clock::now();
	h_output = fann_run(network, h_input);
	stop = std::chrono::high_resolution_clock::now();

	time_elapsed = stop - start;
	std::cout << "Temps de calcul : " << time_elapsed.count() << std::endl;
	// Création des variables devices

	fann_type* d_input, * d_output,* d_weigth;

	cudaMalloc((void**)&d_output, LAYER_MAX_SIZE * sizeof(fann_type));
	cudaMalloc((void**)&d_input, LAYER_MAX_SIZE * sizeof(fann_type));
	cudaMalloc((void**)&d_weigth, LAYER_MAX_SIZE * LAYER_MAX_SIZE * sizeof(fann_type));

	cudaMemcpyAsync(d_input, h_input, sizeof(fann_type) * INPUT, cudaMemcpyHostToDevice,0);

	unsigned int first_weight_pos = 0;
	unsigned int input_size;
	unsigned int output_size;
	
	h_weight = fann_get_layer_weights(network, first_weight_pos, fann_get_total_connections(network));
	cudaMemcpy(d_weigth, h_weight, sizeof(fann_type) * fann_get_total_connections(network), cudaMemcpyHostToDevice);

	start = std::chrono::high_resolution_clock::now();
	for (int layer = 0; layer < NB_LAYER-1; ++layer) {
		input_size = network_prop[layer];
		output_size = network_prop[layer + 1];

		cuda_fann_run<<<1,output_size>>>(d_input, d_output, d_weigth,input_size, output_size);
		
		// Update input value into the next layer
		cudaMemcpyAsync(d_input, d_output, sizeof(fann_type) * LAYER_MAX_SIZE, cudaMemcpyDeviceToDevice,0);
	}
	stop = std::chrono::high_resolution_clock::now();

	// Copy the value of the output into host memory
	cudaMemcpy(h_output, d_output, sizeof(fann_type) * OUTPUT, cudaMemcpyDeviceToHost);



	std::cout << "Resultat : ";

	for (int i = 0; i < OUTPUT; ++i) {
		std::cout << h_output[i] << " " << std::endl;
	}
	time_elapsed = stop - start;
	
	std::cout << "Time : " << time_elapsed.count() << std::endl;


	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_weigth);
	fann_destroy(network);

	return EXIT_SUCCESS;
}