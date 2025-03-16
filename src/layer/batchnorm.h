#pragma once
#ifndef LAYER_BATCHNORM_H
#define LAYER_BATCHNORM_H

#include<stdbool.h>
#include<string.h>

/*卷积结构体-浮点版本,可参考nn.Conv2d*/
struct FloatBatchNorm2d
{
	// Parameters===========================
	int num_features;

	// Variables============================
	float* weight;        // 形状:(num_features)
	float* bias;          // 形状:(num_features)
	float* running_mean;  // 形状:(num_features)
	float* running_var;   // 形状:(num_features)
	float* a_data_ptr;    // 形状:(num_features)
	float* b_data_ptr;    // 形状:(num_features)
}FloatBatchNorm2d;


void batchnorm2d_forward(struct FloatTensor* input, struct FloatBatchNorm2d* layer, struct FloatTensor* output, float* scratchbuf);

#endif