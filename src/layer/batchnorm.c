
#include "tensor.h"
#include "batchnorm.h"

/********************************************************
* Function name : batchnorm2d_forward
* Description   : BN层前向推理：y=nn.BatchNorm2d(x)
* Parameter     :
* @input(FloatTensor*)       输入结构体指针
* @layer(FloatBatchNorm2d)   BN层结构体指针
* @output(FloatTensor*)      输出结构体指针
* @scratchbuf(float*)        scratchbuf指针
* Return        :       无
**********************************************************/
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