#include "sgd_top.h"

void dot_product(SGD_PARAM_CONFIG param,
		int64& mem_addr,
		CacheLine mem_data,
		hls::stream<CacheLine>&  CacheLineFifo,
		int32 cur_sample,
		int32 Q,
		int32 x)
{


	for(int k = 0;k < 32;k++)
	{
#pragma HLS pipeline
		mem_addr = param.addr_a + (cur_sample<<6) + k;
		CacheLineFifo.write(mem_data);
		for(int i = 0;i < BANK; i++)
		{
#pragma HLS unroll factor=8
			for(int n = 0;n<64;n++)
			{
				Q[i] = Q[i] + ((k<param.number_of_bits)? mem_data[i*64+n]*(x[cur_sample+n]>>k) : 0);
			}
		}
	}
}

void gradient(SGD_PARAM_CONFIG param,
		hls::stream<CacheLine>&  CacheLineFifo,
		int32 scale,
		int32 G)
{
	CacheLine a_data;
	for(int k = 0;k < 32;k++)
	{
#pragma HLS pipeline
		CacheLineFifo.read(a_data);
		for(int i = 0;i < BANK; i++)
		{
#pragma HLS unroll factor=8
			for(int n = 0;n<64;n++)
			{
				G[k+n] = G[k+n] + ((k<param.number_of_bits)? a_data[i*64+n]*(scale[i]>>k) : 0);
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


	int a_dot_x;
	int32* scale;
	int32* x ;
	int32 Q[BANK] ;
	int32* G ;
	int32 sample_index;

	for(int e = 1;e < param.number_of_epochs; e++){
		for(int i = 0;i < param.number_of_samples; i = i + param.mini_batch_size)
		{
			for(int n = 0; n < param.dimension; n++)
			{
				G[n] = 0;                              //G[n]初始化
			}

			for(int j = 0;j < param.mini_batch_size;j = j + BANK)
			{
				sample_index = i + j;
				for(int k = 0;k < param.dimension;k = k + 64)
				{
					dot_product(param,*mem_addr,mem_data,CacheLineFifo,sample_index,*Q,*x);
				}
				*mem_addr = param.addr_b + (sample_index>>1) ;
				for(int m = 0;m < BANK; m++)
				{
					scale[m] =  sample_index[0] ? (param.learning_rate * (Q[m] - mem_data(32*(m+1)-1,32*m))) : (param.learning_rate * (Q[m] - mem_data(32*(m+1)+255,32*m+256))) ;//计算每个参数的rate*(h(a)-b)
				}
				for(int k = 0;k < param.dimension;k = k + 64)
				{
					gradient(param,CacheLineFifo,*scale,*G);
				}

			}
			for(int i = 0;i < param.dimension;i++)
			{
				x[i] = x[i] - G[i]/param.mini_batch_size;//更新线性回归的参数x(i) = x(i)- 1/B*rate*J'(x(i))
			}

		}
	}
}
