#pragma once


#define LOG_PATH "../src/debug/"

enum ARRAYTYPE
{
	ARRAY_FLOAT=4,
	ARRAY_CHAR=1,
};

void cnt_update();

int read_data_bin(void* data, char* filename, int data_size, int data_len);

int log_array(void* data, char* filename, int data_size, int data_len, char* str_name);