from io import FileIO
import torch
from torch import float16, nn
import numpy as np
from xor import xor

dtype_tbl = {
        torch.float32 : "float",
}

def _iterate_tensor_print(file, tensor: torch.Tensor, is_vector=False):
    if is_vector:
        for i in range(tensor.shape[0]):
            file.write(f'{tensor[i].item()}')
            if i < tensor.shape[0] - 1:
                file.write(', ')
        file.write('};\n')
    else:
        for i in range(tensor.shape[0]):
            file.write('\t{')
            for j in range(tensor.shape[1]):
                file.write(f'{tensor[i][j].item()}')
                if j < tensor.shape[1] - 1:
                    file.write(', ')
            file.write('}')
            if i < tensor.shape[0] - 1:
                file.write(',\n')
        file.write('\n};\n')


def _implement_forward(source_file, function_declaration, weight_arrays, weight_rows, weight_cols, bias_vectors, bias_sizes, implement_af, item_dtype):
    source_file.write(f'{function_declaration}\n{{\n')

    # Assuming the input to the first layer is named 'input'
    previous_layer_output = 'input'

    for i, (wt, wr, wc, bt, bs) in enumerate(zip(weight_arrays, weight_rows, weight_cols, bias_vectors, bias_sizes)):
        # Create code for matrix multiplication and addition
        output_array_name = f'layer_{i}_output'
        source_file.write(f'    {dtype_tbl[item_dtype]} {output_array_name}[{bs}] = {{0}};\n')
        source_file.write('    for (int i = 0; i < ' + str(bs) + '; i++) {\n')
        source_file.write('        for (int j = 0; j < ' + previous_layer_output + '_size; j++) {\n')
        source_file.write(f'            {output_array_name}[i] += {previous_layer_output}[j] * {wt}[i][j];\n')
        source_file.write('        }\n')
        source_file.write(f'        {output_array_name}[i] += {bt}[i];\n')

        # Apply activation function if needed
        if implement_af:
            source_file.write(f'        {output_array_name}[i] = sigmoid({output_array_name}[i]);\n')

        source_file.write('    }\n\n')
        previous_layer_output = output_array_name

        # Update size variable for next layer's input
        if i < len(weight_arrays) - 1:  # No need for size variable for the last layer
            source_file.write(f'    int {output_array_name}_size = {bs};\n')

    # The last layer's output is the final output
    source_file.write(f'    for (int i = 0; i < {bs}; i++) {{\n')
    source_file.write(f'        output[i] = {previous_layer_output}[i];\n')
    source_file.write('    }\n')
    source_file.write('}\n')

def _tensor_to_c_array(source_file, header_file, layer: nn.Linear, layer_name: str, layer_idx: int):
    # output weights
    item_data_type = layer.weight.dtype
    lrows, lcols = layer.weight.size()
    weight_array_template = f'{layer_name}_model_layer_{layer_idx}_weights'
    weight_array_row_sz_template = f'{layer_name}_model_layer_{layer_idx}_weights_nrows'
    weight_array_col_sz_template = f'{layer_name}_model_layer_{layer_idx}_weights_ncols'

    # write weights array
    source_file.write(f'const {dtype_tbl[item_data_type]} {weight_array_template}[{lrows}][{lcols}] = {{\n')
    _iterate_tensor_print(file=source_file, tensor=layer.weight)

    # write weight size variables
    source_file.write(f'const unsigned int {weight_array_row_sz_template} = {lrows};\n')
    source_file.write(f'const unsigned int {weight_array_col_sz_template} = {lcols};\n\n')
    
    # write weight arrays & vars to header file
    header_file.write(f'extern const {dtype_tbl[item_data_type]} {weight_array_template}[{lrows}][{lcols}];\n')
    header_file.write(f'extern const unsigned int {weight_array_row_sz_template};\n')
    header_file.write(f'extern const unsigned int {weight_array_col_sz_template};\n\n')

    # output biases
    bsize = layer.bias.size()[0]
    bias_array_template = f'{layer_name}_model_layer_{layer_idx}_bias'
    bias_array_sz_template = f'{layer_name}_model_layer_{layer_idx}_bias_size'

    # write bias array
    source_file.write(f'const {dtype_tbl[item_data_type]} {bias_array_template}[{bsize}] = {{ ')
    _iterate_tensor_print(file=source_file, tensor=layer.bias, is_vector=True)

    # write bias size variable
    source_file.write(f'const unsigned int {bias_array_sz_template} = {bsize};\n\n')

    # write bias vector & vars to header file
    header_file.write(f'extern const {dtype_tbl[item_data_type]} {bias_array_template}[{bsize}];\n')
    header_file.write(f'extern const unsigned int {bias_array_sz_template};\n\n')

    return item_data_type, lrows, lcols, bsize, weight_array_template, weight_array_row_sz_template, weight_array_col_sz_template, bias_array_template, bias_array_sz_template

def torch2c(model_pth: str, filename='model', implement_af=False, implement_fwdp=False):
    header_file = open(f'{filename}.h', "a") # dont forget to close
    source_file = open(f'{filename}.c', "a")
    model = torch.load(model_pth)

    header_file.write(f'#ifndef {filename.capitalize()}_H\n#define {filename.capitalize()}_H\n\n')
    source_file.write(f'#include "{filename}.h"\n\n')

    layer_idx = 0
    num_inputs = 0
    num_outputs = 0
    item_dtype = float16

    if implement_fwdp:
        weight_arrays = []
        weight_rows = []
        weight_cols = []
        bias_vectors = []
        bias_size = []
 
    for model_idx, (name, module) in enumerate(model.named_children()):
        # Could be a linear layer
        if isinstance(module, nn.Sequential):
            # iterating through all layers
            for idx, layer in enumerate(module):
                print(layer)
                # could be activation function
                if isinstance(layer, nn.Linear):
                    item_dtype, lrows, lcols, bsize, wt, wr, wc, bt, bs = _tensor_to_c_array(source_file=source_file, header_file=header_file, layer=layer, layer_name=name, layer_idx=layer_idx)

                    if implement_fwdp:
                        weight_arrays.append(wt)
                        weight_rows.append(wr)
                        weight_cols.append(wc)
                        bias_vectors.append(bt)
                        bias_size.append(bs)

                    if layer_idx == 0:
                       num_inputs = lrows

                    num_outputs = bsize
                    layer_idx += 1
                # leakyrelu function case
                elif isinstance(layer, nn.LeakyReLU):
                    print("Leaky Relu")
                # relu case
                elif isinstance(layer, nn.ReLU):
                    print("Relu")
            layer_idx = 0
            break
                    
        # linear layer case
        elif isinstance(module, nn.Linear):
            break
            lrows, lcols = module.weight.size()
            bsize = module.bias.size()[0]
            file.write(f'{dtype_tbl[module.weight.dtype]} {name}_linear_layer_weights[{lrows}][{lcols}] \n')
            file.write(f'{dtype_tbl[module.bias.dtype]} {name}_linear_layer_bias[{bsize}] \n')
            print(module)

    source_file.write(f'unsigned int input_size = {num_inputs};\n\n')

    if implement_fwdp:
        function_decloration = f'void forward({dtype_tbl[item_dtype]} input[{num_inputs}], {dtype_tbl[item_dtype]} output[{num_outputs}])'
        header_file.write(f'{function_decloration};\n\n')
        _implement_forward(source_file=source_file, function_declaration=function_decloration, weight_arrays=weight_arrays, weight_rows=weight_rows, weight_cols=weight_cols, bias_vectors=bias_vectors, bias_sizes=bias_size, implement_af=True, item_dtype=item_dtype)


    header_file.write(f'#endif\n')
    header_file.close()
    source_file.close()
                                                                                                    
torch2c(model_pth='xor.pth', filename='xor', implement_af=False, implement_fwdp=True)
