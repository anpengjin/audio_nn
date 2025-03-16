#include<stdio.h>
#include<stdlib.h>
#include <string.h>

#include "debug.h"

static int cnt = 0;

void cnt_update()
{
    cnt += 1;
}

int read_data_bin(void* data, char* filename, int data_size, int data_len)
{
    char full_path[1000] = "";
    // 拼接路径和文件名
    sprintf(full_path, "%s%s", LOG_PATH, filename);
    // sprintf(full_path, filename);

    FILE* file;
    // 打开二进制文件
    file = fopen(full_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "无法打开文件 %s\n", filename);
        return -1;
    }

    // 读取文件内容到数组
    fread(data, data_size, data_len, file);
    return 0;
}

int log_array(void* data, char* filename, int data_size, int data_len, char* str_name)
{
    char full_path[1000] = "";
    // 拼接路径和文件名
    sprintf(full_path, "%s%s", LOG_PATH, filename);

    FILE* file;
    // 打开二进制文件
    if (cnt == 0) {
        file = fopen(full_path, "w");
    } else {
        file = fopen(full_path, "a");
    }

    if (file == NULL) {
        fprintf(stderr, "无法打开文件 %s\n", filename);
        return -1;
    }

    fprintf(file, "frame=%d\n", cnt);
    if (data_size == ARRAY_FLOAT) {
        float* tensor = (float*)data;
        for (int i = 0; i < data_len; i++) {
            fprintf(file, "%s[%-5d]=%-20.6f", str_name, i, tensor[i]);
            if ((i + 1) % 3 == 0) {
                fprintf(file, "\n");
            }
        }
    }

    // 关闭文件
    fclose(file);

    return 0;
}