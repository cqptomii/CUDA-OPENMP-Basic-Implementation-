
## Calculate network output

input : array of neuron value on the n layer

input_size : size of the input array

output : array of neuron value on th n+1 layer

output_size : size of the output array

weigths : matrix of weigths between layer n and n+1


````
    for each layer in the network do
        fann_kernel_run<<<1,output_size>>>(input,output,weigths,);
        input <- output;
    end for
````

## Calculate one layer activation

In this function every thread Calculate one neuron value on the n+1 layer

idX : index of the current thread

````
    fonction fann_kernel_run(input,output,weigths,input_size,output_size)
        start
            idX <- Threadidx.x

            shared_output[output_size]
            shared_input[input_size]

           shared_input <- input

            __syncthreads()

            for each n in layer(output) do 
                sum <- 0
                for each n1 in layer(input) do
                    sum += weigths[ n1 * output_size + n] * n
                end for

                shared_output[idX] <- activation(sum)

            end for

            __syncthreads

            output[idx] <- shared_output[idX]
        end
````