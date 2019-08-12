#include <ap_int.h>
#define BANK 8
typedef ap_fixed<64,64> int64;
typedef ap_fixed<512,512> CacheLine;
struct SGD_PARAM_CONFIG {
	  int64      addr_a;                       //8  63:0
	  int64      addr_b;                       //8  127:64
	  int64      addr_model;                   //8  191:128

      unsigned int  mini_batch_size;         //4       223:192
      unsigned int  step_size;               //4  //8  255:224
      unsigned int  number_of_epochs;        //4       287:256
      unsigned int  dimension;               //4  //8  319:288
      unsigned int  number_of_samples;       //4       351:320
      unsigned int  number_of_bits;          //4  //8  383:352

      unsigned int  learning_rate;

};
