#pragma once
#ifndef LAYER_NET_H
#define LAYER_NET_H

#include<stdbool.h>
#include<string.h>

#include "tensor.h"
#include "abs.h"
#include "linear.h"
#include "conv.h"
#include "batchnorm.h"
#include "leakyrelu.h"

/*Layer结构体-浮点版本,可参考nn.Conv2d*/
struct FloatNetModule
{
	struct FloatConv2d conv1_conv2d;
	struct FloatBatchNorm2d conv1_bn2d;
	struct FloatLeakyReLU conv1_leakyrelu;
	struct FloatLinear linear1;
	struct FloatLinear linear2;

}FloatNetModule;

int get_ainr_size();

void ainr_init(struct FloatNetModule* net_module);


#endif