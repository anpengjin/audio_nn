// audio_main.c: 定义应用程序的入口点。
//
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<math.h>

#include "tensor.h"
#include "net.h"
#include "abs.h"
#include "linear.h"
#include "conv.h"
#include "batchnorm.h"
#include "leakyrelu.h"

#include "debug.h"


void test_abs()
{
	int b = 1;
	int c = 2;
	int t = 5;
	int f = 10;
	int size = b * c * t * f;

	float* input = (float*)malloc(size * sizeof(float));
	float* output = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < size; i++) {
		input[i] = -(float)i;
		printf("%f ", input[i]);
	}
	printf("\n===============abs====================\n");

	abs_forward(input, size, output);

	for (int i = 0; i < size; i++) {
		printf("%f ", output[i]);
	}
}

void test_linear()
{
	int b = 1;
	int c = 2;
	int t = 2;
	int in_features = 256;
	int out_features = 512;

	float* input = (float*)malloc(b * c * t * in_features * sizeof(float));
	float* output = (float*)malloc(b * c * t * out_features * sizeof(float));

	//for (int i = 0; i < b * c * t * in_features; i++) {
	//	input[i] = -(float)i;
	//	printf("%f ", input[i]);
	//}

	struct FloatLinear float_linear;
	float_linear.in_features = in_features;
	float_linear.out_features = out_features;
	float_linear.bias_flag = true;
	float* weight = (float*)malloc(float_linear.in_features * float_linear.out_features * sizeof(float));
	float* bias = (float*)malloc(float_linear.out_features * sizeof(float));
	for (int i = 0; i < float_linear.in_features * float_linear.out_features; i++) {
		weight[i] = i;
	}
	for (int i = 0; i < float_linear.out_features; i++) {
		bias[i] = i + 1;
	}
	float_linear.weight = weight;
	float_linear.bias = bias;
	printf("\n===============abs====================\n");

	clock_t start, end;
	double cpu_time_used;
	start = clock();
	for (int i = 0; i < 100; i++) {
		linear_forward(input, b * c * t * in_features, &float_linear, output, b * c * t * out_features);
	}
	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
	printf("Time used: %f ms\n", cpu_time_used);

	//for (int i = 0; i < b * c * t * out_features; i++) {
	//	printf("%f ", output[i]);
	//}
}

void test_debug()
{
	int input_size = 1 * 1 * 1 * 257 * 2;
	float* input = (float*)malloc(input_size * sizeof(float));

	read_data_bin((void*)input, "gruc_input_python.bin", ARRAY_FLOAT, input_size);

	log_array((void*)input, "gruc_input_c.log", ARRAY_FLOAT, input_size, "input");
}

void test_gruc_origin()
{
	int input_size = 1 * 1 * 1 * 257 * 2;
	float* input = (float*)malloc(input_size * sizeof(float));

	read_data_bin((void*)input, "gruc_input_python.bin", ARRAY_FLOAT, input_size);
	// log_array((void*)input, "gruc_input_c.log", ARRAY_FLOAT, input_size, "input");

	float* scratchbuf = (float*)malloc(10000 * 257 * 2 * sizeof(float));
	// 1. 求幅度谱
	float* input_mag = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 1 * 1 * 257);
	for (int i = 0; i < 257; i++) {
		input_mag[i] = sqrt(input[2 * i] * input[2 * i] + input[2 * i + 1] * input[2 * i + 1]);
	}

	// 2. self.conv1-》nn.Conv2d
	float* conv1_conv2d_out = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 2 * 1 * 257);

	struct FloatConv2d float_conv2d;
	float_conv2d.in_channels = 1;
	float_conv2d.out_channels = 2;
	float_conv2d.kernel_h = 1;
	float_conv2d.kernel_w = 1;
	float_conv2d.stride_h = 1;
	float_conv2d.stride_w = 1;
	float_conv2d.padding_h = 0;
	float_conv2d.padding_w = 0;
	float_conv2d.dilation_h = 1;
	float_conv2d.dilation_w = 1;
	float_conv2d.groups = 1;
	float_conv2d.bias_flag = true;

	int weight_len = float_conv2d.out_channels * float_conv2d.in_channels / float_conv2d.groups * float_conv2d.kernel_h * float_conv2d.kernel_w;
	float* weight = (float*)malloc(weight_len * sizeof(float));
	float* bias = (float*)malloc(float_conv2d.out_channels * sizeof(float));
	
	read_data_bin((void*)weight, "conv1.0.weight.bin", ARRAY_FLOAT, weight_len);
	read_data_bin((void*)bias, "conv1.0.bias.bin", ARRAY_FLOAT, float_conv2d.out_channels);

	log_array((void*)weight, "conv1.0.weight_c.log", ARRAY_FLOAT, weight_len, "a");
	log_array((void*)bias, "conv1.0.bias_c.log", ARRAY_FLOAT, float_conv2d.out_channels, "a");

	float_conv2d.weight = weight;
	float_conv2d.bias = bias;

	struct FloatTensor conv2d_input;
	conv2d_input.data = input_mag;
	conv2d_input.c = 1;
	conv2d_input.h = 1;
	conv2d_input.w = 257;

	struct FloatTensor conv2d_out;
	conv2d_out.data = conv1_conv2d_out;
	conv2d_out.c = 2;
	conv2d_out.h = 1;
	conv2d_out.w = 257;

	conv2d_forward(&conv2d_input, &float_conv2d, &conv2d_out, scratchbuf);

	log_array((void*)conv2d_out.data, "conv1.0.out.log", ARRAY_FLOAT, 1 * 2 * 1 * 257, "a");

	// 2. self.conv1-》nn.BatchNorm2d
	float* conv1_bn_out = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 2 * 1 * 257);
	struct FloatTensor bn_output;
	bn_output.data = conv1_bn_out;
	bn_output.c = 2;
	bn_output.h = 1;
	bn_output.w = 257;

	float* bn_weight = (float*)scratchbuf; // 2
	scratchbuf += 2;
	float* bn_bias = (float*)scratchbuf;   // 2
	scratchbuf += 2;
	float* bn_running_mean = (float*)scratchbuf; // 2
	scratchbuf += 2;
	float* bn_running_var = (float*)scratchbuf;  // 2
	scratchbuf += 2;

	read_data_bin((void*)bn_weight, "conv1.1.weight.bin", ARRAY_FLOAT, 2);
	read_data_bin((void*)bn_bias, "conv1.1.bias.bin", ARRAY_FLOAT, 2);
	read_data_bin((void*)bn_running_mean, "conv1.1.running_mean.bin", ARRAY_FLOAT, 2);
	read_data_bin((void*)bn_running_var, "conv1.1.running_var.bin", ARRAY_FLOAT, 2);

	float* a_data_ptr = (float*)scratchbuf;  // 2
	scratchbuf += 2;
	float* b_data_ptr = (float*)scratchbuf;  // 2
	scratchbuf += 2;
	for (int i = 0; i < 2; i++){
		float sqrt_var = sqrt(bn_running_var[i] + 0.00001);
		a_data_ptr[i] = bn_weight[i] / sqrt_var;
		b_data_ptr[i] = bn_bias[i] - bn_weight[i] * bn_running_mean[i] / sqrt_var;
	}

	struct FloatBatchNorm2d float_batchnorm2d;
	float_batchnorm2d.num_features = 2;
	float_batchnorm2d.a_data_ptr = a_data_ptr;
	float_batchnorm2d.b_data_ptr = b_data_ptr;

	batchnorm2d_forward(&conv2d_out, &float_batchnorm2d, &bn_output, scratchbuf);

	log_array((void*)bn_output.data, "conv1.1.out.log", ARRAY_FLOAT, 1 * 2 * 1 * 257, "a");


	// 3. self.conv1-》nn.LeakyReLU
	float* conv1_relu_out = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 2 * 1 * 257);
	struct FloatTensor relu_output;
	relu_output.data = conv1_relu_out;
	relu_output.c = 2;
	relu_output.h = 1;
	relu_output.w = 257;

	struct FloatLeakyReLU leaky_relu;
	leaky_relu.negative_slope = 0.01;

	leakyrelu_forward(&bn_output, &leaky_relu, &relu_output, scratchbuf);

	log_array((void*)relu_output.data, "conv1.2.out.log", ARRAY_FLOAT, 1 * 2 * 1 * 257, "a");


}

void test_gruc()
{
	// 1. get_ainr_size =============================================================
	int size = get_ainr_size();
	
	// 2. ainr_init =============================================================
	struct FloatNetModule* net_module = (struct FloatNetModule*)malloc(size);
	ainr_init(net_module);

	int input_size = 1 * 1 * 1 * 257 * 2;
	float* input = (float*)malloc(input_size * sizeof(float));

	read_data_bin((void*)input, "gruc_input_python.bin", ARRAY_FLOAT, input_size);
	// log_array((void*)input, "gruc_input_c.log", ARRAY_FLOAT, input_size, "input");

	float* scratchbuf = (float*)malloc(10000 * 257 * 2 * sizeof(float));
	// 1. 求幅度谱
	float* input_mag = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 1 * 1 * 257);
	for (int i = 0; i < 257; i++) {
		input_mag[i] = sqrt(input[2 * i] * input[2 * i] + input[2 * i + 1] * input[2 * i + 1]);
	}

	// 2. self.conv1-》nn.Conv2d
	struct FloatTensor conv2d_input;
	conv2d_input.data = input_mag;
	conv2d_input.c = 1;
	conv2d_input.h = 1;
	conv2d_input.w = 257;

	float* conv1_conv2d_out = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 2 * 1 * 257);
	struct FloatTensor conv2d_out;
	conv2d_out.data = conv1_conv2d_out;
	conv2d_out.c = 2;
	conv2d_out.h = 1;
	conv2d_out.w = 257;

	conv2d_forward(&conv2d_input, &net_module->conv1_conv2d, &conv2d_out, scratchbuf);

	log_array((void*)conv2d_out.data, "conv1.0.out.log", ARRAY_FLOAT, 1 * 2 * 1 * 257, "a");

	// 2. self.conv1-》nn.BatchNorm2d =======================================================================
	float* conv1_bn_out = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 2 * 1 * 257);
	struct FloatTensor bn_output;
	bn_output.data = conv1_bn_out;
	bn_output.c = 2;
	bn_output.h = 1;
	bn_output.w = 257;

	batchnorm2d_forward(&conv2d_out, &net_module->conv1_bn2d, &bn_output, scratchbuf);

	log_array((void*)bn_output.data, "conv1.1.out.log", ARRAY_FLOAT, 1 * 2 * 1 * 257, "a");


	//// 3. self.conv1-》nn.LeakyReLU
	float* conv1_relu_out = (float*)scratchbuf; // 1 * 1 * 1 * 257
	scratchbuf += (1 * 2 * 1 * 257);
	struct FloatTensor relu_output;
	relu_output.data = conv1_relu_out;
	relu_output.c = 2;
	relu_output.h = 1;
	relu_output.w = 257;

	leakyrelu_forward(&bn_output, &net_module->conv1_leakyrelu, &relu_output, scratchbuf);

	log_array((void*)relu_output.data, "conv1.2.out.log", ARRAY_FLOAT, 1 * 2 * 1 * 257, "a");
}


int main()
{
	// test_abs();
	// test_linear();
	// test_debug();
	test_gruc();
	return 0;
}
