# torch2c
A tool designed to convert pytorch models into c arrays. 

```torch2c [model.pth] [options]...```
| **_Option_**   | **_Function_**                       |
|----------------|--------------------------------------|
| -o, --out      | Output file name                     |
| -a, --act-func | Implement activation functions       |
| -f, --fwd      | Implement forward propagate function |
