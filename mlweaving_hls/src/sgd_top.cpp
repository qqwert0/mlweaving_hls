#include "sgd_top.h"
void rd_mem(SGD_PARAM_CONFIG param,
		int* mem_addr,
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
	static int dimension_algin = 0;



	switch (RdMemFsmState)
	{
	case IDLE:
		if(start){
			RdMemFsmState = START;
			epochs_index = 0;
			batch_index = 0;
			sample_index = 0;
			dimension_index = 0;
			bits_index = 0;
		}
		break;
	case START:
		RdMemFsmState = READ_B;
		dimension_algin = (param.dimension%64 == 0)? param.dimension :(param.dimension/64 + 1)*64;
		*mem_addr = param.addr_b + (sample_index>>4);
		break;
	case READ_B:
		if(!BDataFifo.full()){
			if(sample_index(3,3))
				BDataFifo.write(mem_data(511,256));
			else
				BDataFifo.write(mem_data(255,0));
			RdMemFsmState = READ_A;
			*mem_addr = param.addr_a + (sample_index*dimension_algin/16) + dimension_index/2 + bits_index;
		}
		else
			RdMemFsmState = READ_B;
		break;
	case READ_A:
		if(!ADataFifo.full()){
			ADataFifo.write(mem_data);
			bits_index++;
			if(bits_index == param.number_of_bits){
				RdMemFsmState = READ_A;
				bits_index = 0;
				dimension_index = dimension_index + 64;
				*mem_addr = param.addr_a + (sample_index*dimension_algin/16) + dimension_index/2 + bits_index;
				if(dimension_index >= param.dimension){
					dimension_index =0;
					sample_index = sample_index + BANK;
					if(sample_index >= param.number_of_samples){
						RdMemFsmState = EPOCH_END;
					}
					else{
						RdMemFsmState = READ_B;
						*mem_addr = param.addr_b + (sample_index>>4);
					}
				}
			}
			else{
				*mem_addr = param.addr_a + (sample_index*dimension_algin/16) + dimension_index/2 + bits_index;
				RdMemFsmState = READ_A;
			}

		}
		else{
			RdMemFsmState = READ_A;
		}
		break;
	case EPOCH_END:
		if(epochs_index == param.number_of_epochs-1){
			RdMemFsmState = FSM_END;
			epochs_index = 0;
			sample_index = 0;
		}
		else{
			RdMemFsmState = READ_B;
			sample_index = 0;
			*mem_addr = param.addr_b + (sample_index>>4);
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
		ap_uint<32> x[512][64],
		int &sample_index)
{
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
#pragma HLS pipeline

	enum DotProductFsmStateType {IDLE, START, CALC, JUDGE, EPOCH_END, FSM_END};
	static DotProductFsmStateType DotProductState = IDLE;
	static int epochs_index;
	static int batch_index;
	//static int sample_index;
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
			Q(i*32+31,i*32) = Q(i*32+31,i*32) + ADataTemp[i*64+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			bit_index++;
			if(bit_index == 64){
				DotProductState=JUDGE;
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
				DotProductState = START;
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
		if(epochs_index == param.number_of_epochs-1){
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

	enum GradientFsmStateType {IDLE, READ_SCALE, READ_A, CALC, JUDGE, EPOCH_END, FSM_END};
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
	switch (GradientState)
	{
	case IDLE:
		if(start){
			GradientState = READ_SCALE;
			epochs_index = 0;
			batch_index = 0;
			sample_index = 0;
			dimension_index = 0;
			bits_index = 0;
			bit_index = 0;
		}
		break;
	case READ_SCALE:
		if(!scaleFifo.empty()){
			scaleFifo.read(scale);
			bits_index++;
			GradientState=READ_A;
		}
		else
			GradientState = READ_SCALE;
		break;
	case READ_A:
		if(!A2DataFifo.empty()){
			A2DataFifo.read(ADataTemp);
			bits_index++;
			GradientState=CALC;
		}
		else
			GradientState = READ_A;
		break;
	case CALC:
		for(int i=0;i< BANK;i++){
#pragma HLS unroll factor=8
			if(bits_index == 1){
				G[bit_index] = ADataTemp[i*64+bit_index]*(scale[i]>>bits_index) ;
			}
			else{
				G[bit_index] = G[bit_index] +  ADataTemp[i*64+bit_index]*(scale[i]>>bits_index) ;
			}
		}
		if(bits_index == param.number_of_bits){
			GFifo.write(G[bit_index]);
		}
		bit_index++;
		if(bit_index == 64){
			GradientState=JUDGE;
			bit_index = 0;
		}
		else{
			GradientState=CALC;
		}
		break;
	case JUDGE:
		if(bits_index == param.number_of_bits){
			bits_index = 0;
			dimension_index = dimension_index + 64;
			GradientState = READ_A;
			if(dimension_index >= param.dimension){
				sample_index = sample_index + BANK;
				batch_index = batch_index + BANK;
				dimension_index =0;
				GradientState = READ_SCALE;
				if(batch_index >= param.mini_batch_size){
					batch_index = 0;
				}
				if(sample_index >= param.number_of_samples){
					GradientState = EPOCH_END;
				}
			}
		}
		else{
			GradientState = READ_A;
		}
		break;
	case EPOCH_END:
		if(epochs_index == param.number_of_epochs-1){
			GradientState = FSM_END;
			epochs_index = 0;
			sample_index = 0;
		}
		else{
			GradientState = READ_SCALE;
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
	ap_uint<32> x_updata[512][64],
	stream<ap_uint<32> >& XupdataFifo
	){
#pragma HLS pipeline
//#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
//#pragma HLS ARRAY_RESHAPE variable=x_updata complete dim=2
#pragma HLS dependence variable=x_updata inter false
#pragma HLS dependence variable=x inter false
	//static int wr_dimension_index = 0;
	static int rd_dimension_index = 0;
	static int wr_dimension_index = 0;
	static int epochs_index;
	static int batch_index = 0;
	//static int sample_index;
	static ap_uint<32> x_updata_temp1;
	static ap_uint<32> x_updata_temp2;
	static ap_uint<32> G;
	if(!GFifo.empty()){
		x_updata_temp1 = x_updata[rd_dimension_index/64][rd_dimension_index%64];
		rd_dimension_index++;
		GFifo.read(G);
		x_updata_temp2 = x_updata_temp1 - G/param.mini_batch_size;
		x_updata[wr_dimension_index/64][wr_dimension_index%64] = x_updata_temp2;
		if(batch_index + BANK >= param.mini_batch_size){
			XupdataFifo.write(x_updata_temp2);
		}
		wr_dimension_index = rd_dimension_index;
		if(rd_dimension_index == param.dimension){
			rd_dimension_index = 0;
			wr_dimension_index = 0;
			batch_index = batch_index + BANK;
			if(batch_index >= param.mini_batch_size){
				batch_index = 0;
			}
		}
	}
}

void wr_x(SGD_PARAM_CONFIG param,
		ap_uint<32> x[512][64],
		stream<ap_uint<32> >& XupdataFifo,
		int& mem_wr_addr,
		CacheLine* mem_wr_data,
		bool start,
		bool& done
		){
#pragma HLS pipeline
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
//#pragma HLS dependence variable=x inter false
	enum WrxFsmStateType {IDLE, START, UPDATA_X, JUDGE, EPOCH_END, FSM_END};
	static WrxFsmStateType WrxState = IDLE;
	static int epochs_index;
	static int batch_index;
	static int sample_index;
	static int dimension_index;
	static ap_uint<32> x_updata_temp;
	switch (WrxState)
	{
	case IDLE:
		if(start){
			WrxState = START;
			epochs_index = 0;
			sample_index = 0;
			dimension_index = 0;
			done = 0;
		}
		break;
	case START:
		WrxState = UPDATA_X;
		break;
	case UPDATA_X:
		if(!XupdataFifo.empty()){
			XupdataFifo.read(x_updata_temp);
			x[dimension_index/64][dimension_index%64] = x_updata_temp;
			if((epochs_index == param.number_of_epochs-1) && (sample_index >= param.number_of_samples-BANK)){
				mem_wr_addr = param.addr_model + dimension_index/16;
				int temp = dimension_index%16;
				(*mem_wr_data).range(temp*32+31,temp*32) = x_updata_temp;
			}
			dimension_index++;
			if(dimension_index >= param.dimension){
				dimension_index = 0;
				sample_index = sample_index + BANK;
				WrxState=JUDGE;
			}
			else{
				WrxState=UPDATA_X;
			}
		}
		else{
			WrxState = UPDATA_X;
		}
		break;
	case JUDGE:
		if(sample_index >= param.number_of_samples){
			sample_index = 0;
			WrxState=EPOCH_END;
		}
		else{
			WrxState=START;
		}
		break;
	case EPOCH_END:
		if(epochs_index == param.number_of_epochs-1){
			WrxState = FSM_END;
			epochs_index = 0;
		}
		else{
			WrxState = START;
			epochs_index++;
		}
		break;
	case FSM_END:
		done = 1;
		WrxState = IDLE;
		break;
	}
}



void sgd_top(SGD_PARAM_CONFIG param,int* mem_rd_addr,CacheLine mem_rd_data,int* mem_wr_addr,CacheLine* mem_wr_data,bool start,int* sample_index,bool* done)
{
#pragma HLS INTERFACE ap_ovld port=mem_rd_addr
#pragma HLS INTERFACE ap_vld port=param
#pragma HLS INTERFACE ap_vld port=mem_rd_data
#pragma HLS INTERFACE ap_ovld port=mem_wr_addr
#pragma HLS INTERFACE ap_ovld port=mem_wr_data
#pragma HLS INTERFACE ap_vld port=start
#pragma HLS INTERFACE ap_ovld port=done
#pragma HLS INTERFACE ap_ovld port=sample_index
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


	  static hls::stream<ap_uint<32> >     XupdataFifo("XupdataFifo");
	  #pragma HLS STREAM variable=XupdataFifo depth=4
//	  #pragma HLS DATA_PACK variable=XupdataFifo

	static ap_uint<32> x[512][64];
	static ap_uint<32> x_updata[512][64];

#pragma HLS RESOURCE variable=x core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=x_updata core=RAM_2P_BRAM
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
#pragma HLS ARRAY_RESHAPE variable=x_updata complete dim=2


	rd_mem(param,mem_rd_addr,mem_rd_data,BDataFifo,ADataFifo,start);

	dot_product(param,ADataFifo,A2DataFifo,QFifo,start,x,*sample_index);

	serial_loss(param,BDataFifo,QFifo,scaleFifo);

	gradient(param,scaleFifo,A2DataFifo,GFifo,start);

	updata_x(param,GFifo,x_updata,XupdataFifo);

	wr_x(param,x,XupdataFifo,*mem_wr_addr,mem_wr_data,start,*done);

}
