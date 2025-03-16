#ifndef LAYER_LINEAR_H
#define LAYER_LINEAR_H


#include<stdbool.h>
#include<string.h>

/*���Բ�ṹ��-����汾,�ɲο�nn.Linear*/
struct FloatLinear
{
	// Parameters===========================
	int in_features;
	int out_features;
	bool bias_flag;   // �Ƿ���bias

	// Variables============================
	float* weight;    // ��״:(out_features,in_features)
	float* bias;      // ��״:(out_features)
}FloatLinear;


/********************************************************
* Function name : linear_forward
* Description   : ���Բ�ǰ������y=nn.Linear(x)
* Parameter     :
* @input(float*)      ��������ָ��
* @input_size(int)    ���������С
* @layer(FloatLinear) ���Բ�ṹ��ָ��
* @output(float*)     �������ָ��
* @output_size(int)   ��������С
* Return        :     ��
**********************************************************/
void linear_forward(float* input, int input_size, struct FloatLinear* layer, float* output, int output_size);


#endif