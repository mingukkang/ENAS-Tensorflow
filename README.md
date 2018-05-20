## ENAS-Tensorflow

I will explain the code of Efficient Neural Architecture Search(ENAS), especially micro search case.

And unlike the author's code, I rewrite the code of ENAS  for windows user!

Before running the code, you should unpack the attacged data as shown below.

![사진1](https://github.com/MINGUKKANG/ENAS-Tensorflow/blob/master/images/unpack.PNG)
<br/>You can train Controller of ENAS with the following short code:
```
python main_controller_child_trainer.py
```
<br/>After finishing,   you can train the child network with the following code:
```
python main_child_trainer.py -n child_fixed_arc "0 1 1 0 0 1 2 4 2 4 1 0 1 1 4 3 2 2 3 4 1 1 1 2 0 1 0 3 1 3 0 3 1 2 3 3 1 2 5 0"
```

**The string in the above code like "0 1 1 0 0 1 2 ~ " is the result of main_controller_child.py**

**The first 14 numbers are for the architecture for conv layers, and the rest are for pooling layers**

## Enviroment
- OS: Window 10(Ubuntu 16.04 is possible)

- Python 3.5

- Tensorflow-gpu version:  1.4.0rc2 

- OpenCV 3.4.1


## References
**Paper: https://arxiv.org/abs/1802.03268**

**Autors' implementation: https://github.com/melodyguan/enas**

**Data Pipeline: https://github.com/MINGUKKANG/MNIST-Tensorflow-Code**

## License
All rights related to this code are reserved to the author of ENAS

(Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean)
