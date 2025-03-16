#pragma once
#ifndef LAYER_ABS_H
#define LAYER_ABS_H


/********************************************************
* Function name : abs_forward
* Description   : 取绝对值层前向推理：y=abs(x)
* Parameter     :
* @input(float*)      输入数组指针
* @input_size(int)    输入数组大小
* @output(float*)     输出数组指针
* Return        :     无
**********************************************************/
void abs_forward(float* input, int input_size, float* output);

#endif