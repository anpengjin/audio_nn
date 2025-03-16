
#include "tensor.h"
#include "conv.h"


void conv2d_forward(struct FloatTensor* input, struct FloatConv2d* layer, struct FloatTensor* output, float* scratchbuf)
{
    int kernel_h = layer->kernel_h;
    int kernel_w = layer->kernel_w;
    int in_channels = layer->in_channels;
    int out_channels = layer->out_channels;
    int stride_h = layer->stride_h;
    int stride_w = layer->stride_w;
    int padding_h = layer->padding_h;
    int padding_w = layer->padding_w;
    int dilation_h = layer->dilation_h;
    int dilation_w = layer->dilation_w;

    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    int outh = (input->h - kernel_extent_h) / stride_h + 1;
    int outw = (input->w - kernel_extent_w) / stride_w + 1;

    const float* weight_ptr = layer->weight;
    const float* bias_ptr = layer->bias;

    // kernel offset ===============================================================
    int* space_ofs = (int*)scratchbuf;
    scratchbuf += kernel_h * kernel_w;
    {
        int p1 = 0;
        int p2 = 0;
        int gap = kernel_w * dilation_w - kernel_extent_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    const int maxk = kernel_h * kernel_w;


    // calculate ===============================================================
    for (int p = 0; p < out_channels; p++) {
        float* outptr = &output->data[p * outh * outw];
        for (int i = 0; i < outh; i++) {
            for (int j = 0; j < outw; j++) {
                float sum = 0.f;
                if (layer->bias_flag) sum = bias_ptr[p];
                const float* kptr = weight_ptr + p * in_channels * maxk;
                // channels
                for (int q = 0; q < in_channels; q++)
                {
                    const float* sptr = input->data + i * input->w * stride_w + j * stride_w;
                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                    }
                    kptr += maxk;
                }
                outptr[j] = sum;
            }
            outptr += outw;
        }
    }
}