
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fann.h>
#include <iostream>
#include <ostream>
#include <vector>

#define INPUT 200
#define OUTPUT 100
#define HIDE_LAYER 10
#define NB_LAYER HIDE_LAYER+2
#define LAYER_MAX_SIZE 500

__global__ void cuda_fann_run(float *input,float* output, float* weigths, int idInput,int idOutput){
	__shared__ float s_input[LAYER_MAX_SIZE];
	__shared__ float s_output[LAYER_MAX_SIZE];

	int idX = threadIdx.x;
	if (idX < idInput) {
		s_input[idX] = input[idX];
	}
	
	//Synchronize allocation of shared space
	__syncthreads();

	//Reset the sum
	float sum = 0.0f;
	
	if (idX < idOutput) {
		for (int i = 0; i < idInput; ++i) {
			if (idInput < idOutput) {
				sum += s_input[i] * weigths[i * idOutput + idX];
			}
		}

		s_output[idX] = 1.0f / (1.0f + expf(-sum)); // activation sigmoÏde
	}

	__syncthreads();

	//Copy shared output value in global memory
	if (idX < idOutput) {
		output[idX] = s_output[idX];
	}
}

struct fann* create_neural_network(std::vector<unsigned int> value) {

	if (value.size() < 2) {
		std::cerr << "ERROR :: A NETWORK NEED TO HAVE AT LEAST 2 neuron layers" << std::endl;
		return nullptr;
	}

	struct fann* temp = fann_create_standard(value.size(), value.data());

	fann_set_activation_function_hidden(temp, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(temp, FANN_SIGMOID_SYMMETRIC);

	return temp;
}

float* fann_get_layer_weights(struct fann* network, int Input_layer_index) {
	int layer_amount = fann_get_num_layers(network);
	if (Input_layer_index > layer_amount) {
		std::cerr << "Error input layer is not in the network" << std::endl;
		return nullptr;
	}
	unsigned int* layer_sizes;
	fann_get_layer_array(network, layer_sizes);

	unsigned int num_input_neuron = layer_sizes[Input_layer_index];
	unsigned int num_output_neuron = layer_sizes[Input_layer_index+1];

	fann_type* weigths_network;
	fann_get_weights(network, weigths_network);

	unsigned int num_weights = num_input_neuron * num_output_neuron;

	unsigned int first_weight_index = 0;

	for (int i = 0; i < Input_layer_index; ++i) {
		first_weight_index += layer_sizes[i]*layer_sizes[i+1];
	}
	
	float* weigths_matrix = new float[num_weights];

	for (unsigned int i = 0; i < num_weights; ++i) {
		weigths_matrix[i] = weigths_network[first_weight_index + i];
	}

	return weigths_matrix;
}
int main(){
	std::vector<unsigned int> network_properties = { INPUT,400,500,500,500,500,500,500,500,500,500,OUTPUT };

	struct fann* network = create_neural_network(network_properties);
	if ( network == nullptr) {
		std::cerr << "ERROR :: NETWORK CREATION FAILED" << std::endl;
		return EXIT_FAILURE;
	}

	// Ajuste les poids de chaque neurones à chaque itération
	fann_set_training_algorithm(network, FANN_TRAIN_RPROP);
	// Ajuste le taux d'apprentissage des poids
	fann_set_learning_rate(network, 0.5);

	// Entrainement du réseau sur un fichier donnée :

	// Tester le réseau
	fann_type h_input[INPUT] = { 0.5f,0.8f };
	fann_type h_output[OUTPUT];
	

	 float* d_input, * d_output,* d_weigth;

	cudaMalloc((void**)&d_output, LAYER_MAX_SIZE * sizeof(float));
	cudaMalloc((void**)&d_input, LAYER_MAX_SIZE * sizeof(float));
	cudaMalloc((void**)&d_weigth, LAYER_MAX_SIZE * LAYER_MAX_SIZE * sizeof(float));

	for (int layer = 0; layer < NB_LAYER; ++layer) {
		int input_size = network_properties[layer];
		int output_size = network_properties[layer + 1];
		
		cudaMemcpy(d_input, h_input, sizeof(float) * LAYER_MAX_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(d_weigth, fann_get_layer_weights(network, layer),sizeof(float)*input_size, cudaMemcpyHostToDevice);

		cuda_fann_run<<<1,output_size>>>(d_input, d_output, d_weigth, input_size, output_size);
		
		// Update input value into the next layer
		cudaMemcpy(d_input, d_output, sizeof(float) * output_size, cudaMemcpyDeviceToDevice);
	}

	// Copy the value of the output into host memory

	cudaMemcpy(h_output, d_output, sizeof(float) * OUTPUT, cudaMemcpyDeviceToHost);

	std::cout << "Resultat : ";

	for (int i = 0; i < OUTPUT; ++i) {
		std::cout << d_output[i] << " ";
	}

	fann_destroy(network);
	return EXIT_SUCCESS;
}