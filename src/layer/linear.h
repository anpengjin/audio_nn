#ifndef LAYER_LINEAR_H
#define LAYER_LINEAR_H


#include<stdbool.h>
#include<string.h>

/*线性层结构体-浮点版本,可参考nn.Linear*/
struct FloatLinear
{
	// Parameters===========================
	int in_features;
	int out_features;
	bool bias_flag;   // 是否有bias

	// Variables============================
	float* weight;    // 形状:(out_features,in_features)
	float* bias;      // 形状:(out_features)
}FloatLinear;


/********************************************************
* Function name : linear_forward
* Description   : 线性层前向推理：y=nn.Linear(x)
* Parameter     :
* @input(float*)      输入数组指针
* @input_size(int)    输入数组大小
* @layer(FloatLinear) 线性层结构体指针
* @output(float*)     输出数组指针
* @output_size(int)   输出数组大小
* Return        :     无
**********************************************************/
void linear_forward(float* input, int input_size, struct FloatLinear* layer, float* output, int output_size);


#endif