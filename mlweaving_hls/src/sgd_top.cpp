#include "sgd_top.h"
void rd_mem(SGD_PARAM_CONFIG param,
		int64& mem_addr,
		CacheLine mem_data,
		stream<ap_uint<256> >&  BDataFifo,
		stream<ap_uint<512> >&  ADataFifo,
		bool start)
{
#pragma HLS pipeline
	enum RdMemFsmStateType {IDLE, START, READ_B, READ_A, EPOCH_END, FSM_END};
	static RdMemFsmStateType RdMemFsmState = IDLE;
	static int epochs_index;
	static int batch_index;
	static ap_uint<32> sample_index;
	static int dimension_index;
	static int bits_index;

	switch (RdMemFsmState)
	{
	case IDLE:
		if(start)
			RdMemFsmState = START;
		break;
	case START:
		RdMemFsmState = READ_B;
		break;
	case READ_B:
		if(!BDataFifo.full()){
			mem_addr = param.addr_b + (sample_index<<6);
			if(sample_index%2)
				BDataFifo.write(mem_data(511,256));
			else
				BDataFifo.write(mem_data(255,0));
			sample_index = sample_index + BANK;
			RdMemFsmState = READ_A;
		}
		else
			RdMemFsmState = READ_B;
		break;
	case READ_A:
		if(!ADataFifo.full()){
			mem_addr = param.addr_a + (sample_index>>1);
			ADataFifo.write(mem_data);
			bits_index++;
			if(bits_index == param.number_of_bits && dimension_index >= param.dimension){
				bits_index = 0;
				dimension_index =0;
				if(sample_index >= param.number_of_samples){
					RdMemFsmState = EPOCH_END;
				}
				else
					RdMemFsmState = READ_B;
			}
			else if(bits_index == param.number_of_bits){
				RdMemFsmState = READ_A;
				bits_index = 0;
				dimension_index = dimension_index + 64;
			}
			else{
				RdMemFsmState = READ_A;
			}

		}
		else{
			RdMemFsmState = READ_A;
		}
		break;
	case EPOCH_END:
		if(epochs_index == param.number_of_epochs){
			RdMemFsmState = FSM_END;
			epochs_index = 0;
			sample_index = 0;
		}
		else{
			RdMemFsmState = READ_B;
			sample_index = 0;
			epochs_index++;
		}
		break;
	case FSM_END:
		RdMemFsmState = IDLE;
		break;

	}
}


void dot_product(SGD_PARAM_CONFIG param,
		stream<ap_uint<512> >&  ADataFifo,
		stream<ap_uint<512> >&  A2DataFifo,
		stream<ap_uint<256> >&  QFifo,
		bool start,
		ap_uint<32> x[512][64])
{
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
#pragma HLS pipeline

	enum DotProductFsmStateType {IDLE, START, CALC, JUDGE, EPOCH_END, FSM_END};
	static DotProductFsmStateType DotProductState = IDLE;
	static int epochs_index;
	static int batch_index;
	static int sample_index;
	static int dimension_index;
	static int bits_index;
	static int bit_index;
	static ap_uint<256> Q;
	ap_uint<512> ADataTemp;

	switch (DotProductState)
	{
	case IDLE:
		if(start){
			DotProductState = START;
			epochs_index = 0;
			batch_index = 0;
			sample_index = 0;
			dimension_index = 0;
			bits_index = 0;
			bit_index = 0;
		}
		break;
	case START:
		bit_index = 0;
		if(!ADataFifo.empty() && !A2DataFifo.full()){
			ADataFifo.read(ADataTemp);
			A2DataFifo.write(ADataTemp);
			bits_index++;
			DotProductState=CALC;
		}
		else
			DotProductState = START;
		break;
	case CALC:
		for(int i=0;i< BANK;i++){
#pragma HLS unroll factor=8
			Q(i*32+31,i*32) = Q(i*32+31,i*32) + ADataTemp[i*64+bit_index] * x[dimension_index/64][bit_index];
			bit_index++;
			if(bit_index == 64){
				DotProductState=START;
			}
			else{
				DotProductState=CALC;
			}
		}
		break;
	case JUDGE:
		if(bits_index == param.number_of_bits){
			bits_index = 0;
			dimension_index = dimension_index + 64;
			DotProductState = START;
			if(dimension_index >= param.dimension){
				sample_index = sample_index + BANK;
				dimension_index =0;
				QFifo.write(Q);
				Q = 0;
				if(sample_index >= param.number_of_samples){
					DotProductState = EPOCH_END;
				}
			}
		}
		else{
			DotProductState = START;
		}
		break;
	case EPOCH_END:
		if(epochs_index == param.number_of_epochs){
			DotProductState = FSM_END;
			epochs_index = 0;
			sample_index = 0;
		}
		else{
			DotProductState = START;
			sample_index = 0;
			epochs_index++;
		}
		break;
	case FSM_END:
		DotProductState = IDLE;
		break;
	}
}



void serial_loss(SGD_PARAM_CONFIG param,
		stream<ap_uint<256> >&  BDataFifo,
		stream<ap_uint<256> >&  QFifo,
		stream<ap_uint<256> >&  scaleFifo){
#pragma HLS pipeline
	static ap_uint<256> b;
	static ap_uint<256> Q;
	static ap_uint<256> scale;
	if(!BDataFifo.empty() && !QFifo.empty()){
		BDataFifo.read(b);
		QFifo.read(Q);
		for(int i = 0;i < BANK; i++){
#pragma HLS unroll factor=8
			scale(i*32+31,i*32) =  param.learning_rate * (Q(i*32+31,i*32) - b(i*32+31,i*32)) ;//计算每个参数的rate*(h(a)-b)
		}
		scaleFifo.write(scale);
	}
}

void gradient(SGD_PARAM_CONFIG param,
		stream<ap_uint<256> >&  scaleFifo,
		stream<ap_uint<512> >&  A2DataFifo,
		stream<ap_uint<32> >&  GFifo,
		bool start
		){

#pragma HLS pipeline

	enum GradientFsmStateType {IDLE, START, CALC, JUDGE, EPOCH_END, FSM_END};
	static GradientFsmStateType GradientState = IDLE;
	static int epochs_index;
	static int batch_index;
	static int sample_index;
	static int dimension_index;
	static int bits_index;
	static int bit_index;
	ap_uint<512> ADataTemp;
	ap_uint<256> scale;
	static ap_uint<32> G[64];
#pragma HLS ARRAY_RESHAPE variable=G complete dim=1
	switch (GradientState)
	{
	case IDLE:
		if(start){
			GradientState = START;
			epochs_index = 0;
			batch_index = 0;
			sample_index = 0;
			dimension_index = 0;
			bits_index = 0;
			bit_index = 0;
		}
		break;
	case START:
		if(!A2DataFifo.empty() && !scaleFifo.empty()){
			A2DataFifo.read(ADataTemp);
			scaleFifo.read(scale);
			bits_index++;
			GradientState=CALC;
		}
		else
			GradientState = START;
		break;
	case CALC:
		for(int i=0;i< BANK;i++){
#pragma HLS unroll factor=8
			if(bits_index == param.number_of_bits){
				G[bit_index] = ADataTemp[i*64+bit_index]*(scale[i]>>bits_index) ;
			}
			else{
				G[bit_index] = G[bit_index] +  ADataTemp[i*64+bit_index]*(scale[i]>>bits_index) ;
			}
		}
		if(bit_index == 64){
			GradientState=JUDGE;
			bit_index = 0;
		}
		else{
			GradientState=CALC;
		}
		if(bits_index == param.number_of_bits){
			GFifo.write(G[bit_index]);
		}
		bit_index++;
		break;
	case JUDGE:
		if(bits_index == param.number_of_bits){
			bits_index = 0;
			dimension_index = dimension_index + 64;
			GradientState = START;
			if(dimension_index >= param.dimension){
				sample_index = sample_index + BANK;
				batch_index = batch_index + BANK;
				dimension_index =0;
				if(batch_index >= param.mini_batch_size){
					batch_index = 0;
				}
				if(sample_index >= param.number_of_samples){
					GradientState = EPOCH_END;
				}
			}
		}
		else{
			GradientState = START;
		}
		break;
	case EPOCH_END:
		if(epochs_index == param.number_of_epochs){
			GradientState = FSM_END;
			epochs_index = 0;
			sample_index = 0;
		}
		else{
			GradientState = START;
			sample_index = 0;
			epochs_index++;
		}
		break;
	case FSM_END:
		GradientState = IDLE;
		break;
	}
}

void updata_x(SGD_PARAM_CONFIG param,
	stream<ap_uint<32> >&  GFifo,
	ap_uint<32> x[512][64],
	ap_uint<32> x_updata[512][64]
	){
#pragma HLS pipeline
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
#pragma HLS ARRAY_RESHAPE variable=x_updata complete dim=2
	static int dimension_index = 0;
	static ap_uint<32> G;
	if(!GFifo.empty()){
		GFifo.read(G);
		x_updata[dimension_index/64][dimension_index%64] = x[dimension_index/64][dimension_index%64] - G/param.mini_batch_size;
		x[dimension_index/64][dimension_index%64] = x_updata[dimension_index/64][dimension_index%64];
		dimension_index ++;
		if(dimension_index == param.dimension){
			dimension_index = 0;
		}
	}
}

/*

void dot_product(SGD_PARAM_CONFIG param,
		int64& mem_addr,
		CacheLine mem_data,
		hls::stream<CacheLine>&  CacheLineFifo,
		int Q[BANK],
		ap_uint<512> x[512][4])
{
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2

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
*/

void sgd_top(SGD_PARAM_CONFIG param,int64* mem_addr,CacheLine mem_data,bool start)
{
#pragma HLS INTERFACE ap_ovld port=mem_addr
#pragma HLS INTERFACE ap_vld port=param
#pragma HLS INTERFACE ap_vld port=mem_data
#pragma HLS DATAFLOW

	  static hls::stream<ap_uint<512> >     ADataFifo("ADataFifo");
	  #pragma HLS STREAM variable=ADataFifo depth=4
	  //#pragma HLS DATA_PACK variable=CacheLineFifo
	  static hls::stream<ap_uint<512> >     A2DataFifo("A2DataFifo");
	  #pragma HLS STREAM variable=A2DataFifo depth=4
	  static hls::stream<ap_uint<256> >     BDataFifo("BDataFifo");
	  #pragma HLS STREAM variable=BDataFifo depth=4
	  static hls::stream<ap_uint<256> >     QFifo("QFifo");
	  #pragma HLS STREAM variable=QFifo depth=4
	  static hls::stream<ap_uint<256> >     scaleFifo("scaleFifo");
	  #pragma HLS STREAM variable=scaleFifo depth=4
	  static hls::stream<ap_uint<32> >     GFifo("GFifo");
	  #pragma HLS STREAM variable=scaleFifo depth=4


	static ap_uint<32> x[512][64];
	static ap_uint<32> x_updata[512][64];

#pragma HLS RESOURCE variable=x core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=x_updata core=RAM_2P_BRAM
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
#pragma HLS ARRAY_RESHAPE variable=x_updata complete dim=2


	rd_mem( param,*mem_addr,mem_data,BDataFifo,ADataFifo,start);

	dot_product(param,ADataFifo,A2DataFifo,QFifo,start,&x[64]);

	serial_loss(param,BDataFifo,QFifo,scaleFifo);

	gradient(param,scaleFifo,A2DataFifo,GFifo,start);

	updata_x(param,GFifo,&x[64],&x_updata[64]);
}
