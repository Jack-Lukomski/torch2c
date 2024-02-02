#ifndef Xor_H
#define Xor_H

extern const float layers_model_layer_0_weights[2][2];
extern const unsigned int layers_model_layer_0_weights_nrows;
extern const unsigned int layers_model_layer_0_weights_ncols;

extern const float layers_model_layer_0_bias[2];
extern const unsigned int layers_model_layer_0_bias_size;

extern const float layers_model_layer_1_weights[2][2];
extern const unsigned int layers_model_layer_1_weights_nrows;
extern const unsigned int layers_model_layer_1_weights_ncols;

extern const float layers_model_layer_1_bias[2];
extern const unsigned int layers_model_layer_1_bias_size;

extern const float layers_model_layer_2_weights[1][2];
extern const unsigned int layers_model_layer_2_weights_nrows;
extern const unsigned int layers_model_layer_2_weights_ncols;

extern const float layers_model_layer_2_bias[1];
extern const unsigned int layers_model_layer_2_bias_size;

void forward(float input[2], float output[1]);

#endif
