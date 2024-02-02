#include "xor.h"
#include <stdio.h>
#include <math.h>

const float layers_model_layer_0_weights[2][2] = {
	{-9.675437927246094, 7.735860824584961},
	{5.0653839111328125, -6.453698635101318}
};
const unsigned int layers_model_layer_0_weights_nrows = 2;
const unsigned int layers_model_layer_0_weights_ncols = 2;

const float layers_model_layer_0_bias[2] = { -4.918869972229004, -2.331406831741333};
const unsigned int layers_model_layer_0_bias_size = 2;

const float layers_model_layer_1_weights[2][2] = {
	{-6.460925579071045, -4.316792964935303},
	{-6.454933166503906, -4.5563273429870605}
};
const unsigned int layers_model_layer_1_weights_nrows = 2;
const unsigned int layers_model_layer_1_weights_ncols = 2;

const float layers_model_layer_1_bias[2] = { 1.365173101425171, 1.4766753911972046};
const unsigned int layers_model_layer_1_bias_size = 2;

const float layers_model_layer_2_weights[1][2] = {
	{-3.7841715812683105, -4.214906215667725}
};
const unsigned int layers_model_layer_2_weights_nrows = 1;
const unsigned int layers_model_layer_2_weights_ncols = 2;

const float layers_model_layer_2_bias[1] = { 2.8522331714630127};
const unsigned int layers_model_layer_2_bias_size = 1;

unsigned int input_size = 2;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void forward(float input[2], float output[1])
{
    float layer_0_output[layers_model_layer_0_bias_size] = {0};
    for (int i = 0; i < layers_model_layer_0_bias_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer_0_output[i] += input[j] * layers_model_layer_0_weights[i][j];
        }
        layer_0_output[i] += layers_model_layer_0_bias[i];
        layer_0_output[i] = sigmoid(layer_0_output[i]);
    }

    int layer_0_output_size = layers_model_layer_0_bias_size;
    float layer_1_output[layers_model_layer_1_bias_size] = {0};
    for (int i = 0; i < layers_model_layer_1_bias_size; i++) {
        for (int j = 0; j < layer_0_output_size; j++) {
            layer_1_output[i] += layer_0_output[j] * layers_model_layer_1_weights[i][j];
        }
        layer_1_output[i] += layers_model_layer_1_bias[i];
        layer_1_output[i] = sigmoid(layer_1_output[i]);
    }

    int layer_1_output_size = layers_model_layer_1_bias_size;
    float layer_2_output[layers_model_layer_2_bias_size] = {0};
    for (int i = 0; i < layers_model_layer_2_bias_size; i++) {
        for (int j = 0; j < layer_1_output_size; j++) {
            layer_2_output[i] += layer_1_output[j] * layers_model_layer_2_weights[i][j];
        }
        layer_2_output[i] += layers_model_layer_2_bias[i];
        layer_2_output[i] = sigmoid(layer_2_output[i]);
    }

    for (int i = 0; i < layers_model_layer_2_bias_size; i++) {
        output[i] = layer_2_output[i];
    }
}

int main(void)
{
    float input[2];
    float output[1];

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            input[0] = (float) i;
            input[1] = (float) j;
            forward(input, output);

            printf("%f ^ %f = %f\n", input[0], input[1], output[0]);
        }
    }

    return 0;
}
