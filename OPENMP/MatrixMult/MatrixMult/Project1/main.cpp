#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>

#define N 2000
#define M 2000
#define P 1000

void init(double* mat_a, double* mat_b) {
	for (int i = 0; i < P; ++i) {
		for (int j = 0; j < N; ++j) {
			mat_a[j*P + i] = (double)rand() / RAND_MAX;
		}
		for (int k = 0; k < M; ++k) {
			mat_b[i * M + k] = (double)rand() / RAND_MAX;
		}
	}
}
int main(int argc, char* argv[]) {

	std::chrono::duration<double> time_elapsed;
	std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;

	// Set the amount of thread in the program
	if (argc > 1) {
		int thread_amount = atoi(argv[1]);
		omp_set_num_threads(thread_amount);
	}else
		omp_set_num_threads(omp_get_max_threads());


	double* mat_a, * mat_b, * mat_result;

	mat_a = new double[N*P];
	mat_b = new double[P * M];
	mat_result = new double[N * M];

	// init matrix values
	init(mat_a, mat_b);
	
	start = std::chrono::high_resolution_clock::now();

	// Matrix multiplication
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M;++j) {
			mat_result[i * M + j] = 0.0;
			for (int k = 0; k < P; ++k) {
				mat_result[i * M + j] += mat_a[i * P + k] * mat_b[k * M + j];
			}
		}
	}

	stop = std::chrono::high_resolution_clock::now();

	time_elapsed = stop - start;

	std::cout << "Matrix multiplication in " << time_elapsed.count() << "  seconds" << std::endl;

	delete[] mat_a;
	delete[] mat_b;
	delete[] mat_result;

	return EXIT_SUCCESS;
}