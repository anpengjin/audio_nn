#pragma once
#ifndef LAYER_TENSOR_H
#define LAYER_TENSOR_H


#include<stdbool.h>
#include<string.h>

/*����tensor-����汾,�ɲο�nn.Tensor*/
struct FloatTensor
{
	// Parameters===========================
	int c;
	int h;
	int w;

	// Variables============================
	float* data;    // ��״:(1, C, H, W)
}FloatTensor;


#endif