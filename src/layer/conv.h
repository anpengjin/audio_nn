#pragma once
#ifndef LAYER_CONV_H
#define LAYER_CONV_H


#include<stdbool.h>
#include<string.h>

/*����ṹ��-����汾,�ɲο�nn.Conv2d*/
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
	float* weight;    // ��״:(out_channels, in_channels/groups, kernel_h, kernel_w)
	float* bias;      // ��״:(out_channels)
}FloatConv2d;


/********************************************************
* Function name : conv2d_forward
* Description   : �����ǰ������y=nn.Conv2d(x)
* Parameter     :
* @input(FloatTensor*)  ����ṹ��ָ��
* @layer(FloatConv2d)   �����ṹ��ָ��
* @output(FloatTensor*) ����ṹ��ָ��
* @scratchbuf(float*)   scratchbufָ��
* Return        :       ��
**********************************************************/
void conv2d_forward(struct FloatTensor* input, struct FloatConv2d* layer, struct FloatTensor* output, float* scratchbuf);

#endif