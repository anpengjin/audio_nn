#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#include "debug.h"
#include "net.h"



int get_ainr_size()
{
	int size = sizeof(FloatNetModule) + 8;
	return size;
}

void ainr_init(struct FloatNetModule* net_module)
{
	struct FloatConv2d* float_conv2d = &net_module->conv1_conv2d;
	float_conv2d->in_channels = 1;
	float_conv2d->out_channels = 2;
	float_conv2d->kernel_h = 1;
	float_conv2d->kernel_w = 1;
	float_conv2d->stride_h = 1;
	float_conv2d->stride_w = 1;
	float_conv2d->padding_h = 0;
	float_conv2d->padding_w = 0;
	float_conv2d->dilation_h = 1;
	float_conv2d->dilation_w = 1;
	float_conv2d->groups = 1;
	float_conv2d->bias_flag = true;

	int weight_len = float_conv2d->out_channels * float_conv2d->in_channels / float_conv2d->groups * float_conv2d->kernel_h * float_conv2d->kernel_w;
	float* weight = (float*)malloc(weight_len * sizeof(float));
	float* bias = (float*)malloc(float_conv2d->out_channels * sizeof(float));

	read_data_bin((void*)weight, "conv1.0.weight.bin", ARRAY_FLOAT, weight_len);
	read_data_bin((void*)bias, "conv1.0.bias.bin", ARRAY_FLOAT, float_conv2d->out_channels);
	float_conv2d->weight = weight;
	float_conv2d->bias = bias;
	// ==========================================================================================================
	struct FloatBatchNorm2d* float_batchnorm2d = &net_module->conv1_bn2d;
	float_batchnorm2d->num_features = 2;

	float* bn_weight = (float*)malloc(float_batchnorm2d->num_features * sizeof(float));
	float* bn_bias = (float*)malloc(float_batchnorm2d->num_features * sizeof(float));
	float* bn_running_mean = (float*)malloc(float_batchnorm2d->num_features * sizeof(float));
	float* bn_running_var = (float*)malloc(float_batchnorm2d->num_features * sizeof(float));

	read_data_bin((void*)bn_weight, "conv1.1.weight.bin", ARRAY_FLOAT, 2);
	read_data_bin((void*)bn_bias, "conv1.1.bias.bin", ARRAY_FLOAT, 2);
	read_data_bin((void*)bn_running_mean, "conv1.1.running_mean.bin", ARRAY_FLOAT, 2);
	read_data_bin((void*)bn_running_var, "conv1.1.running_var.bin", ARRAY_FLOAT, 2);

	float* a_data_ptr = (float*)malloc(float_batchnorm2d->num_features * sizeof(float));
	float* b_data_ptr = (float*)malloc(float_batchnorm2d->num_features * sizeof(float));
	for (int i = 0; i < 2; i++) {
		float sqrt_var = sqrt(bn_running_var[i] + 0.00001);
		a_data_ptr[i] = bn_weight[i] / sqrt_var;
		b_data_ptr[i] = bn_bias[i] - bn_weight[i] * bn_running_mean[i] / sqrt_var;
	}

	float_batchnorm2d->a_data_ptr = a_data_ptr;
	float_batchnorm2d->b_data_ptr = b_data_ptr;

	// ==========================================================================================================
	struct FloatLeakyReLU* leaky_relu = &net_module->conv1_leakyrelu;
	leaky_relu->negative_slope = 0.01;
}
