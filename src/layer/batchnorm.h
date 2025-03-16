#pragma once
#ifndef LAYER_BATCHNORM_H
#define LAYER_BATCHNORM_H

#include<stdbool.h>
#include<string.h>

/*����ṹ��-����汾,�ɲο�nn.Conv2d*/
struct FloatBatchNorm2d
{
	// Parameters===========================
	int num_features;

	// Variables============================
	float* weight;        // ��״:(num_features)
	float* bias;          // ��״:(num_features)
	float* running_mean;  // ��״:(num_features)
	float* running_var;   // ��״:(num_features)
	float* a_data_ptr;    // ��״:(num_features)
	float* b_data_ptr;    // ��״:(num_features)
}FloatBatchNorm2d;


void batchnorm2d_forward(struct FloatTensor* input, struct FloatBatchNorm2d* layer, struct FloatTensor* output, float* scratchbuf);

#endif