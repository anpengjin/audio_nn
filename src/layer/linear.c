
#include "linear.h"


void linear_forward(float* input, int input_size, struct FloatLinear* layer, float* output, int output_size)
{
	int in_features = layer->in_features;
	int out_features = layer->out_features;
	int bias_flag = layer->bias_flag;
	float* weight = layer->weight;
	float* bias = layer->bias;

	int batch = input_size / in_features;

	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < out_features; j++) {
			*output = 0;
			for (int k = 0; k < in_features; k++) {
				*output += input[i * in_features + k] * weight[j * in_features + k];
			}
			*output += bias[j];
			output += 1;
		}
	}
}