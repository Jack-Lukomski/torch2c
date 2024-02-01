from io import FileIO
import torch
from torch import nn
import numpy as np
from xor import xor

filename = 'model.c'

file = open(filename, "a")

dtype_tbl = {
        torch.float32 : "float",
}

test = torch.load('xor.pth')

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


def _tensor_to_c_array(file, layer: nn.Linear, layer_name: str, layer_idx: int):
    # output weights
    lrows, lcols = layer.weight.size()
    file.write(f'{dtype_tbl[layer.weight.dtype]} {layer_name}_model_layer_{layer_idx}_weights[{lrows}][{lcols}] = {{\n')
    _iterate_tensor_print(file=file, tensor=layer.weight)
    file.write(f'unsigned int {layer_name}_model_layer_{layer_idx}_weights_nrows = {lrows};\n')
    file.write(f'unsigned int {layer_name}_model_layer_{layer_idx}_weights_ncols = {lcols};\n\n')

    # output biases
    bsize = layer.bias.size()[0]
    file.write(f'{dtype_tbl[layer.bias.dtype]} {layer_name}_model_layer_{layer_idx}_bias[{bsize}] = {{ ')
    _iterate_tensor_print(file=file, tensor=layer.bias, is_vector=True)
    file.write(f'unsigned int {layer_name}_model_layer_{layer_idx}_bias_size = {bsize};\n\n')

layer_idx = 0

for model_idx, (name, module) in enumerate(test.named_children()):
    # Could be a linear layer
    if isinstance(module, nn.Sequential):
        # iterating through all layers
        for idx, layer in enumerate(module):
            print(layer)
            # could be activation function
            if isinstance(layer, nn.Linear):
                _tensor_to_c_array(file=file, layer=layer, layer_name=name, layer_idx=layer_idx)
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

    print("next\n\n\n")

# np.savetxt('test.csv', test['linear_relu_stack.0.weight'], delimiter=',')
file.close()
