#pragma once
#ifndef LAYER_CONV_H
#define LAYER_CONV_H


#include<stdbool.h>
#include<string.h>

/*卷积结构体-浮点版本,可参考nn.Conv2d*/
struct FloatConv2d
{
	// Parameters===========================
	int in_channels;
	int out_channels;
	int kernel_h; 
	int kernel_w;
	int stride_h;
	int stride_w;
	int padding_h;
	int padding_w;
	int dilation_h;
	int dilation_w;
	int groups;
	bool bias_flag;

	// Variables============================
	float* weight;    // 形状:(out_channels, in_channels/groups, kernel_h, kernel_w)
	float* bias;      // 形状:(out_channels)
}FloatConv2d;


/********************************************************
* Function name : conv2d_forward
* Description   : 卷积层前向推理：y=nn.Conv2d(x)
* Parameter     :
* @input(FloatTensor*)  输入结构体指针
* @layer(FloatConv2d)   卷积层结构体指针
* @output(FloatTensor*) 输出结构体指针
* @scratchbuf(float*)   scratchbuf指针
* Return        :       无
**********************************************************/
void conv2d_forward(struct FloatTensor* input, struct FloatConv2d* layer, struct FloatTensor* output, float* scratchbuf);

#endif