
#include "tensor.h"
#include "batchnorm.h"

void batchnorm2d_forward(struct FloatTensor* input, struct FloatBatchNorm2d* layer, struct FloatTensor* output, float* scratchbuf)
{
    int channels = input->c;
    int h = input->h;
    int w = input->w;

    for (int q = 0; q < channels; q++)
    {
        float* ptr = &input->data[q * h * w];
        float* outptr = &output->data[q * h * w];

        float a = layer->a_data_ptr[q];
        float b = layer->b_data_ptr[q];

        for (int i = 0; i < h * w; i++)
        {
            outptr[i] = a * ptr[i] + b;
        }
    }
}