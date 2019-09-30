#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace hls;
#define BANK 8
typedef ap_fixed<32,32> int32;
typedef ap_fixed<64,64> int64;
typedef ap_uint<512> CacheLine;
struct SGD_PARAM_CONFIG {
	int      addr_a;                       //8  63:0
	int      addr_b;                       //8  127:64
	int      addr_model;                   //8  191:128

      unsigned int  mini_batch_size;         //4       223:192
      unsigned int  step_size;               //4  //8  255:224
      unsigned int  number_of_epochs;        //4       287:256
      unsigned int  dimension;               //4  //8  319:288
      unsigned int  number_of_samples;       //4       351:320
      unsigned int  number_of_bits;          //4  //8  383:352

      unsigned int  learning_rate;

};
struct X_UINT {
	ap_uint<32> x[64];
};
void sgd_top(SGD_PARAM_CONFIG param,int* mem_rd_addr,CacheLine mem_rd_data,int* mem_wr_addr,CacheLine* mem_wr_data,bool start,int* sample_index,bool* done,int* count1,int* count2);
