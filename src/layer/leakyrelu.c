
#include "tensor.h"
#include "leakyrelu.h"


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