#pragma once
#ifndef LAYER_LEAKYRELU_H
#define LAYER_LEAKYRELU_H


#include<stdbool.h>
#include<string.h>

/*激活层结构体-浮点版本,可参考nn.LeakyReLU*/
struct FloatLeakyReLU
{
	// Parameters===========================
	float negative_slope;   // Default: 1e-2

}FloatLeakyReLU;


/********************************************************
* Function name : leakyrelu_forward
* Description   : 激活层前向推理：y=nn.LeakyRelu(x)
* Parameter     :
* @input(FloatTensor*)       输入结构体指针
* @layer(FloatLeakyReLU)     激活层结构体指针
* @output(FloatTensor*)      输出结构体指针
* @scratchbuf(float*)        scratchbuf指针
* Return        :            无
**********************************************************/
void leakyrelu_forward(struct FloatTensor* input, struct FloatLeakyReLU* layer, struct FloatTensor* output, float* scratchbuf);

#endif