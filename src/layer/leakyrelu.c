
#include "tensor.h"
#include "leakyrelu.h"

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
void leakyrelu_forward(struct FloatTensor* input, struct FloatLeakyReLU* layer, struct FloatTensor* output, float* scratchbuf)
{
	float negative_slope = layer->negative_slope;
	int size = input->c * input->h * input->w;

	float* sptr = input->data;
	float* outptr = output->data;

	for (int i = 0; i < size; i++) {
		if (sptr[i] < 0) {
			outptr[i] = negative_slope * sptr[i];
		}else {
			outptr[i] = sptr[i];
		}
	}
}