#pragma once
#ifndef LAYER_TENSOR_H
#define LAYER_TENSOR_H


#include<stdbool.h>
#include<string.h>

/*张量tensor-浮点版本,可参考nn.Tensor*/
struct FloatTensor
{
	// Parameters===========================
	int c;
	int h;
	int w;

	// Variables============================
	float* data;    // 形状:(1, C, H, W)
}FloatTensor;


#endif