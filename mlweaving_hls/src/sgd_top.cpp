#include "sgd_top.h"

void dot_product(SGD_PARAM_CONFIG param,
		int64& mem_addr,
		CacheLine mem_data,
		hls::stream<CacheLine>&  CacheLineFifo,
		ap_uint<32> cur_sample,
		int *Q,
		ap_uint<512> x[4])
{
#pragma HLS ARRAY_RESHAPE variable=x complete dim=1

	for(int k = 0;k < param.number_of_bits;k++)
	{
#pragma HLS pipeline
		int temp;
		mem_addr = param.addr_a + (cur_sample<<6) + k;
		CacheLineFifo.write(mem_data);
		for(int i = 0;i < BANK; i++)
		{
#pragma HLS unroll factor=8
			for(int n = 0;n<64;n++)
			{
				temp = x[n/16](((n%16)*32+31),(n%16)*32);
				Q[i] = Q[i] +  mem_data[i*64+n]*(temp>>k);
			}
		}
	}
}

void gradient(SGD_PARAM_CONFIG param,
		hls::stream<CacheLine>&  CacheLineFifo,
		int *scale,
		ap_uint<512> G[4])
{
#pragma HLS ARRAY_RESHAPE variable=G complete dim=1

	CacheLine a_data;
	for(int k = 0;k < param.number_of_bits;k++)
	{
#pragma HLS pipeline
		a_data = CacheLineFifo.read();
		for(int i = 0;i < BANK; i++)
		{
#pragma HLS unroll factor=8
			for(int n = 0;n<64;n++)
			{
				G[n/16]((n%16)*32+31,(n%16)*32) = G[n/16]((n%16)*32+31,(n%16)*32) +  a_data[i*64+n]*(scale[i]>>k) ;
			}
		}
	}
}


void sgd_top(SGD_PARAM_CONFIG param,int64* mem_addr,CacheLine mem_data)
{
#pragma HLS INTERFACE ap_ovld port=mem_addr
#pragma HLS INTERFACE ap_vld port=param
#pragma HLS INTERFACE ap_vld port=mem_data

	  static hls::stream<CacheLine>     CacheLineFifo("CacheLineFifo");
	  #pragma HLS STREAM variable=CacheLineFifo depth=4
	  #pragma HLS DATA_PACK variable=CacheLineFifo


	static int scale[BANK];
	static ap_uint<512> x[512][4];
	static int Q[BANK] ;
	static ap_uint<512> G[512][4];
	static ap_uint<32> sample_index;
	static int b;

//#pragma HLS RESOURCE variable=x core=RAM_2P_BRAM
//#pragma HLS RESOURCE variable=G core=RAM_2P_BRAM


	for(int e = 1;e < param.number_of_epochs; e++){
		for(int i = 0;i < param.number_of_samples; i = i + param.mini_batch_size)
		{
			for(int n = 0; n < param.dimension; n=n+64)
			{
				G[n][0] = 0;G[n][1] = 0;G[n][2] = 0;G[n][3] = 0;                              //G[n]初始化
			}

			for(int j = 0;j < param.mini_batch_size;j = j + BANK)
			{
//#pragma HLS pipeline
				sample_index = i + j;
				for(int k = 0;k < param.dimension;k = k + 64)
				{
					dot_product(param,*mem_addr,mem_data,CacheLineFifo,sample_index,Q,x[k/64]);
				}
				*mem_addr = param.addr_b + (sample_index>>1) ;
				for(int m = 0;m < BANK; m++)
				{
#pragma HLS unroll factor=8
					b = sample_index[0] ? mem_data(32*(m+1)-1,32*m) : mem_data(32*(m+1)+255,32*m+256);
					scale[m] =  param.learning_rate * (Q[m] - b) ;//计算每个参数的rate*(h(a)-b)
				}
				for(int k = 0;k < param.dimension;k = k + 64)
				{
					gradient(param,CacheLineFifo,scale,G[k/64]);
				}

			}
			for(int i = 0;i < param.dimension;i = i+64)
			{
				x[i][0] = x[i][0] - G[i][0];//更新线性回归的参数x(i) = x(i)- 1/B*rate*J'(x(i))
				x[i][1] = x[i][1] - G[i][1];
				x[i][2] = x[i][2] - G[i][2];
				x[i][3] = x[i][3] - G[i][3];
			}

		}
	}
}
