#include<math.h>

#include "abs.h"

void abs_forward(float* input, int input_size, float* output)
{
    for (int i = 0; i < input_size; i++)
    {
        output[i] = abs(input[i]);
    }
}

