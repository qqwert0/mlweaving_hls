#include "sgd_top.h"
void sgd_top(SGD_PARAM_CONFIG param,int64* mem_addr,CacheLine mem_data)
{
#pragma HLS INTERFACE ap_ovld port=mem_addr
#pragma HLS INTERFACE ap_vld port=param
#pragma HLS INTERFACE ap_vld port=mem_data
	int a_dot_x;
	int scale;
	int64* x ;
	int64 Q[BANK] ;
	int64* G ;


	for(int e = 1;e < param.number_of_epochs; e++){
		for(int i = 0;i < param.number_of_samples; i = i + param.mini_batch_size)
		{
			for(int n = 0; n < param.dimension; n++)
			{
				G[n] = 0;                              //G[n]初始化
			}

			for(int j = 0;j < param.mini_batch_size;j = j + BANK)
			{
				for(int m = 0;m < BANK; m++)
				{
					for(int k = 0;k < param.dimension;k = k + 64)
					{
						for(int l = 0;l < param.number_of_bits;l++)
						{
							*mem_addr = param.addr_a + (i+j)/BANK*param.number_of_bits + k/64*param.number_of_bits + l;
							for(int n = 0;n<64;n++)
							{
								Q[m] = Q[m] + mem_data[m*64+n]*(x[k+n]>>l);  //每个sample计算其h(a)=a(0)*x(0)+a(1)*x(1)+...+a(m)*x(m);m个特征
							}

						}
					}
					*mem_addr = param.addr_b + i + j + m;
					scale = param.learning_rate * (Q[m] - mem_data);//计算每个参数的rate*(h(a)-b)
					for(int k = 0;k < param.dimension;k = k + 64)
					{
						for(int l = 0;l < param.number_of_bits;l++)
						{
							*mem_addr = param.addr_a + (i+j)/BANK*param.number_of_bits + k/64*param.number_of_bits + l;
							for(int n = 0;n<64;n++)
							{
								G[k+n] = G[k+n] + mem_data[m*64+n]*(scale>>l);//计算loss fuction导数J'(x(i))=(h(a)-b)*a(i)
							}

						}
					}

				}

			}
			for(int i = 0;i < param.dimension;i++)
			{
				x[i] = x[i] - G[i]/param.mini_batch_size;//更新线性回归的参数x(i) = x(i)- 1/B*rate*J'(x(i))
			}

		}
	}
}
