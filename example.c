#include <math.h>
#include <stdio.h>

float layers_model_layer_0_weights[2][2] = {
	{-9.675437927246094, 7.735860824584961},
	{5.0653839111328125, -6.453698635101318}
};
unsigned int layers_model_layer_0_weights_nrows = 2;
unsigned int layers_model_layer_0_weights_ncols = 2;

float layers_model_layer_0_bias[2] = { -4.918869972229004, -2.331406831741333};
unsigned int layers_model_layer_0_bias_size = 2;

float layers_model_layer_1_weights[2][2] = {
	{-6.460925579071045, -4.316792964935303},
	{-6.454933166503906, -4.5563273429870605}
};
unsigned int layers_model_layer_1_weights_nrows = 2;
unsigned int layers_model_layer_1_weights_ncols = 2;

float layers_model_layer_1_bias[2] = { 1.365173101425171, 1.4766753911972046};
unsigned int layers_model_layer_1_bias_size = 2;

float layers_model_layer_2_weights[1][2] = {
	{-3.7841715812683105, -4.214906215667725}
};
unsigned int layers_model_layer_2_weights_nrows = 1;
unsigned int layers_model_layer_2_weights_ncols = 2;

float layers_model_layer_2_bias[1] = { 2.8522331714630127};
unsigned int layers_model_layer_2_bias_size = 1;

float output[1];

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void forward(unsigned int size, float input[size]) {
    // First Layer Calculations
    float layer_0_output[2] = {0};
    for (int i = 0; i < layers_model_layer_0_weights_nrows; i++) {
        for (int j = 0; j < layers_model_layer_0_weights_ncols; j++) {
            layer_0_output[i] += input[j] * layers_model_layer_0_weights[i][j];
        }
        layer_0_output[i] += layers_model_layer_0_bias[i];
        layer_0_output[i] = sigmoid(layer_0_output[i]);
    }

    // Second Layer Calculations
    float layer_1_output[2] = {0};
    for (int i = 0; i < layers_model_layer_1_weights_nrows; i++) {
        for (int j = 0; j < layers_model_layer_1_weights_ncols; j++) {
            layer_1_output[i] += layer_0_output[j] * layers_model_layer_1_weights[i][j];
        }
        layer_1_output[i] += layers_model_layer_1_bias[i];
        layer_1_output[i] = sigmoid(layer_1_output[i]);
    }

    // Third Layer Calculations
    output[0] = 0;
    for (int j = 0; j < layers_model_layer_2_weights_ncols; j++) {
        output[0] += layer_1_output[j] * layers_model_layer_2_weights[0][j];
    }
    output[0] += layers_model_layer_2_bias[0];
    output[0] = sigmoid(output[0]);
}

int main(void)
{
    float input[2];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            input[0] = (float)i;
            input[1] = (float)j;
            forward(2, input);
            printf("input: %d ^ %d\noutput: %f\n\n", i, j, output[0]);
        }
    }

    return 0;
}
