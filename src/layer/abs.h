#pragma once
#ifndef LAYER_ABS_H
#define LAYER_ABS_H


/********************************************************
* Function name : abs_forward
* Description   : ȡ����ֵ��ǰ������y=abs(x)
* Parameter     :
* @input(float*)      ��������ָ��
* @input_size(int)    ���������С
* @output(float*)     �������ָ��
* Return        :     ��
**********************************************************/
void abs_forward(float* input, int input_size, float* output);

#endif