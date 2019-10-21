#include "sgd_top.h"
void rd_mem(SGD_PARAM_CONFIG param,
		stream<CacheLine> &a_rd_data,
		stream<ap_uint<256> > &b_rd_data,
		stream<ap_uint<256> >&  BDataFifo,
		stream<ap_uint<512> >&  ADataFifo,
		bool start
		)
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
	static ap_uint<256> b_data_temp;
	static CacheLine a_data_temp;



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
			//count1 = 0;
		}
		break;
	case START:
		RdMemFsmState = READ_B;
		dimension_algin = (param.dimension%64 == 0)? param.dimension :(param.dimension/64 + 1)*64;

		break;
	case READ_B:
		if(!BDataFifo.full() && !b_rd_data.empty()){
			b_rd_data.read(b_data_temp);
			BDataFifo.write(b_data_temp);
			RdMemFsmState = READ_A;
			//*count2 = *count2 +1;
			//*count1 = b_data_temp;
		}
		else
			RdMemFsmState = READ_B;
		break;
	case READ_A:
		if(!ADataFifo.full() && !a_rd_data.empty()){
			a_rd_data.read(a_data_temp);
			ADataFifo.write(a_data_temp);

			//*count2 = mem_data(63,32);

			bits_index++;
			if(bits_index == param.number_of_bits){
				RdMemFsmState = READ_A;
				bits_index = 0;
				dimension_index = dimension_index + 64;
				//*mem_addr = param.addr_a + (sample_index*dimension_algin/16) + dimension_index/2 + bits_index;
				if(dimension_index >= param.dimension){
					dimension_index =0;
					sample_index = sample_index + BANK;
					if(sample_index >= param.number_of_samples){
						RdMemFsmState = EPOCH_END;
					}
					else{
						RdMemFsmState = READ_B;
						//*mem_addr = param.addr_b + (sample_index>>4);
					}
				}
			}
			else{
				//*mem_addr = param.addr_a + (sample_index*dimension_algin/16) + dimension_index/2 + bits_index;
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
			//*mem_addr = param.addr_b + (sample_index>>4);
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
		X_UINT x[512] ,
		bool start,
		int &sample_index,
		X_UINT* count1,
		int* count2
		)
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
	static ap_uint<512> ADataTemp;

	//static int Q[8];
//#pragma HLS ARRAY_RESHAPE variable=Q complete dim=1


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
			//count2 = 0;
			//count1 = 0;
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
			//Q[i] = Q[i] + ADataTemp[i*64+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q.range(i*32+31,i*32) = Q.range(i*32+31,i*32) + ADataTemp[i*64+bit_index] * (x[dimension_index/64].x[bit_index]>>bits_index);
			/*Q(31,0) = Q(31,0) + ADataTemp[bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q(63,32) = Q(63,32) + ADataTemp[64+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q(95,64) = Q(95,64) + ADataTemp[128+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q(127,96) = Q(127,96) + ADataTemp[192+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q(159,128) = Q(159,128) + ADataTemp[256+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q(191,160) = Q(191,160) + ADataTemp[320+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q(223,192) = Q(223,192) + ADataTemp[384+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);
			Q(255,224) = Q(255,224) + ADataTemp[448+bit_index] * (x[dimension_index/64][bit_index]>>bits_index);*/
		}
		//*count1 = x[dimension_index/64];
		//*count2 = *count2 + 1;
			bit_index++;
			if(bit_index == 64){
				DotProductState=JUDGE;
			}
			else{
				DotProductState=CALC;
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
				*count2 = *count2 + 1;
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
		stream<ap_uint<256> >&  scaleFifo
		){
#pragma HLS pipeline
	static ap_uint<256> b;
	static ap_uint<256> Q;
	static ap_uint<256> scale;
	static int scale_temp;
	if(!BDataFifo.empty() && !QFifo.empty()){
		BDataFifo.read(b);
		QFifo.read(Q);
		//count2++;
		for(int i = 0;i < BANK; i++){
#pragma HLS unroll factor=8
			scale_temp = (Q(i*32+31,i*32) - b(i*32+31,i*32))>>12;
			scale.range(i*32+31,i*32) =  scale_temp ;//计算每个参数的rate*(h(a)-b)
			 /*scale(31,0) =  param.learning_rate * (Q(31,0) - b(31,0)) ;
			scale(63,32) =  param.learning_rate * (Q(63,32) - b(63,32)) ;
			scale(95,64) =  param.learning_rate * (Q(95,64) - b(95,64)) ;
			scale(127,96) =  param.learning_rate * (Q(127,96) - b(127,96)) ;
			scale(159,128) =  param.learning_rate * (Q(159,128) - b(159,128)) ;
			scale(191,160) =  param.learning_rate * (Q(191,160) - b(191,160)) ;
			scale(223,192) =  param.learning_rate * (Q(223,192) - b(223,192)) ;
			scale(255,224) =  param.learning_rate * (Q(255,224) - b(255,224)) ;*/
		}
		scaleFifo.write(scale);
		//*count1 = scale;
		//*count2 = *count2 + 1;

	}
}

void gradient(SGD_PARAM_CONFIG param,
		stream<ap_uint<256> >&  scaleFifo,
		stream<ap_uint<512> >&  A2DataFifo,
		stream<X_UINT >&  GFifo,
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
	static int bank_index;
	static ap_uint<512> ADataTemp;
	static ap_uint<256> scale;
	static X_UINT G;
	static int scale_temp;
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
			bank_index = 0;
			//*count2 = 0;
		}
		break;
	case READ_SCALE:
		if(!scaleFifo.empty()){
			scaleFifo.read(scale);
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
		for(int i=0;i< 64;i++){
#pragma HLS unroll factor=64
			scale_temp = scale(bank_index*32+31,bank_index*32);
			if((bits_index == 1) && (bank_index == 0)){
				G.x[i] = ADataTemp[bank_index*64+i]*(scale_temp>>bits_index);
				//(*count1).x[i]=AData_Temp[bank_index*64+i];
			}
			//else if(bits_index == param.number_of_bits){
				//G.x[i] = G.x[i] +  ADataTemp[bank_index*64+i]*(scale(bank_index*32+31,bank_index*32)>>bits_index) ;
			//}
			else{
				G.x[i] = G.x[i] +  ADataTemp[bank_index*64+i]*(scale_temp>>bits_index) ;
			}
		}
		//*count2 = *count2 +1;
		//*count1 = G;
		bank_index++;
		if(bank_index == 8){

			GradientState=JUDGE;
			bank_index = 0;
		}
		else{
			GradientState=CALC;
		}
		break;
	case JUDGE:
		if(bits_index == param.number_of_bits){
			bits_index = 0;
			GFifo.write(G);
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
	stream<X_UINT >&  GFifo,
	X_UINT x_updata[512],
	stream<X_UINT >& XupdataFifo
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
	static X_UINT x_updata_temp1;
	static X_UINT x_updata_temp2;
	static X_UINT G;
	if(!GFifo.empty()){
		x_updata_temp1 = x_updata[rd_dimension_index/64];
		GFifo.read(G);
		for(int i=0;i<64;i++){
#pragma HLS unroll factor=64
			x_updata_temp2.x[i] = x_updata_temp1.x[i] - G.x[i]/param.mini_batch_size;
		}
		//*count2 = *count2 +1;
		//*count1 = x_updata_temp2;
		x_updata[rd_dimension_index/64] = x_updata_temp2;
		rd_dimension_index = rd_dimension_index + 64;
		if(batch_index + BANK >= param.mini_batch_size){
			XupdataFifo.write(x_updata_temp2);
			//count1++;
		}
		wr_dimension_index = rd_dimension_index;
		if(rd_dimension_index >= param.dimension){
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
		X_UINT x[512],
		stream<X_UINT> &XupdataFifo,
		//stream<X_UINT> &XFifo,
		stream<X_UINT> &Xupload,
		bool start,
		bool& done
		){
#pragma HLS pipeline
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
#pragma HLS dependence variable=x inter false
	enum WrxFsmStateType {IDLE, START, UPDATA_X, JUDGE, EPOCH_END, FSM_END};
	static WrxFsmStateType WrxState = IDLE;
	static int epochs_index;
	static int batch_index;
	static int sample_index;
	static int dimension_index;
	static int dimension_algin = 0;
	static X_UINT x_updata_temp;
	static X_UINT x_temp;
	switch (WrxState)
	{
	case IDLE:
		if(start){
			WrxState = START;
			epochs_index = 0;
			sample_index = 0;
			batch_index = 0;
			dimension_index = 0;
			//count1 = 0;
			done = 0;
			dimension_algin = (param.dimension%64 == 0)? param.dimension :(param.dimension/64 + 1)*64;
			//for(int i=0;i<64;i++){
			//	x_temp.x[i] = 0;
			//}
		}
		break;
	case START:
		//XFifo.write(x_temp);
		//dimension_index = dimension_index + 64;
		//if(dimension_index >= dimension_algin){
			WrxState = UPDATA_X;
		//}
		//else{
		//	WrxState = START;
		//}
		break;
	case UPDATA_X:
		if(!XupdataFifo.empty()){
			XupdataFifo.read(x_updata_temp);
			x[dimension_index/64] = x_updata_temp;
			//XFifo.write(x_updata_temp);
			if((epochs_index == param.number_of_epochs-1) && (sample_index >= param.number_of_samples-BANK)){
				Xupload.write(x_updata_temp);
			}
			dimension_index = dimension_index + 64;
			if(dimension_index >= dimension_algin){
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
			WrxState=UPDATA_X;
		}
		break;
	case EPOCH_END:
		if(epochs_index == param.number_of_epochs-1){
			WrxState = FSM_END;
			epochs_index = 0;
		}
		else{
			WrxState = UPDATA_X;
			epochs_index++;
		}
		break;
	case FSM_END:
		done = 1;
		WrxState = IDLE;
		break;
	}
}



void sgd_top(SGD_PARAM_CONFIG param,
		stream<CacheLine> &a_rd_data,
		stream<ap_uint<256> > &b_rd_data,
		stream<X_UINT> &Xupload,
		bool start,int* sample_index,bool* done,X_UINT* count1,int* count2)
{
#pragma HLS INTERFACE ap_stable port=param
#pragma HLS INTERFACE ap_stable port=start
#pragma HLS INTERFACE ap_stable port=done
#pragma HLS INTERFACE ap_stable port=sample_index
#pragma HLS INTERFACE ap_stable port=count1
#pragma HLS INTERFACE ap_stable port=count2

#pragma  HLS resource core=AXI4Stream variable=a_rd_data metadata="-bus_bundle a_rd_data"
#pragma HLS DATA_PACK variable=a_rd_data
#pragma  HLS resource core=AXI4Stream variable=b_rd_data metadata="-bus_bundle b_rd_data"
#pragma HLS DATA_PACK variable=b_rd_data
#pragma  HLS resource core=AXI4Stream variable=Xupload metadata="-bus_bundle Xupload"
#pragma HLS DATA_PACK variable=Xupload

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
	  static hls::stream<X_UINT >     GFifo("GFifo");
	  #pragma HLS STREAM variable=GFifo depth=4
	  #pragma HLS DATA_PACK variable=GFifo


	  static hls::stream<X_UINT >     XupdataFifo("XupdataFifo");
	  #pragma HLS STREAM variable=XupdataFifo depth=4
	  #pragma HLS DATA_PACK variable=XupdataFifo

	  //static hls::stream<X_UINT >     XFifo("XFifo");
	  //#pragma HLS STREAM variable=XFifo depth=4
	  //#pragma HLS DATA_PACK variable=XFifo

	//static ap_uint<32> x[512][64];
	//static ap_uint<32> x_updata[512][64];
	static X_UINT x[512] ;
	static X_UINT x_updata[512];

#pragma HLS RESOURCE variable=x core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=x_updata core=RAM_2P_BRAM
#pragma HLS ARRAY_RESHAPE variable=x complete dim=2
#pragma HLS ARRAY_RESHAPE variable=x_updata complete dim=2


	rd_mem(param,a_rd_data,b_rd_data,BDataFifo,ADataFifo,start);

	dot_product(param,ADataFifo,A2DataFifo,QFifo,x,start,*sample_index,count1,count2);

	serial_loss(param,BDataFifo,QFifo,scaleFifo);

	gradient(param,scaleFifo,A2DataFifo,GFifo,start);

	updata_x(param,GFifo,x_updata,XupdataFifo);

	wr_x(param,x,XupdataFifo,Xupload,start,*done);

}
