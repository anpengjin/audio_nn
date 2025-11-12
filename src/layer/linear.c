
#include "linear.h"

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
void linear_forward(float* input, int input_size, struct FloatLinear* layer, float* output, int output_size)
{
	int in_features = layer->in_features;
	int out_features = layer->out_features;
	int bias_flag = layer->bias_flag;
	float* weight = layer->weight;
	float* bias = layer->bias;

	int batch = input_size / in_features;

	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < out_features; j++) {
			*output = 0;
			for (int k = 0; k < in_features; k++) {
				*output += input[i * in_features + k] * weight[j * in_features + k];
			}
			*output += bias[j];
			output += 1;
		}
	}
}