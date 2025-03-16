#pragma once
#ifndef LAYER_LEAKYRELU_H
#define LAYER_LEAKYRELU_H


#include<stdbool.h>
#include<string.h>

/*�����ṹ��-����汾,�ɲο�nn.LeakyReLU*/
struct FloatLeakyReLU
{
	// Parameters===========================
	float negative_slope;   // Default: 1e-2

}FloatLeakyReLU;

void leakyrelu_forward(struct FloatTensor* input, struct FloatLeakyReLU* layer, struct FloatTensor* output, float* scratchbuf);

#endif