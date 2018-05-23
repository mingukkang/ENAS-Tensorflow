## ENAS-Tensorflow

I will explain the code of Efficient Neural Architecture Search(ENAS), especially case of micro search.

Unlike the author's code, This code can work in a windows 10 enviroment and you can use your own data.



## Enviroment
- OS: Window 10(Ubuntu 16.04 is possible)

- Python 3.5

- Tensorflow-gpu version:  1.4.0rc2 

- OpenCV 3.4.1

## How to run

**<br/>At first, you should unpack the attacged data as shown below.**

![사진1](https://github.com/MINGUKKANG/ENAS-Tensorflow/blob/master/images/unpack.PNG)

**<br/> Next, you should rewrite the code below.**

```
Located in main_controller_child_train.py and main_child_trainer.py

DEFINE_string("output_dir",  , "./output")
DEFINE_string("train_data_dir",  , "./data/train")
DEFINE_string("val_data_dir",  , "./data/valid")
DEFINE_string("test_data_dir",  , "./data/test")
DEFINE_integer("channel",1, "MNIST: 1, Cifar10: 3")
```

**<br/>Then, You can train Controller of ENAS with the following short code:**
```
python main_controller_child_trainer.py
```
**<br/>After finishing,   you can train the child network with the following code:**
```
python main_child_trainer.py -n child_fixed_arc "0 0 1 4 0 0 0 3 1 4 0 3 0 0 0 0 0 0 0 2 1 1 0 3 0 1 0 3 1 0 1 1 0 2 1 0 1 0 0 1"
```

The string in the above code like "0 0 1 4 0 0 0 ~ " is the result of main_controller_child.py

The first 20 numbers are for the architecture for conv layers, and the rest are for pooling layers.

## Explained

### 1. Controller

First, we will build the sampler as shown in the picture below.

<br/>![사진2](https://github.com/MINGUKKANG/ENAS-Tensorflow/blob/master/images/Controller_init.png)

<br/>Then we will make controller using sampler's output "next_c_1, next_h_1".

<br/>![사진3](https://github.com/MINGUKKANG/ENAS-Tensorflow/blob/master/images/Controller.PNG)

<br/> After getting the "next_c_5, next_h_5", you must do the following to renew "Anchors,   Anchors_w_1".

<br/>![사진4](https://github.com/MINGUKKANG/ENAS-Tensorflow/blob/master/images/Anchors_appen.PNG)

### 2. Controller_Loss

To enable the Controller to make better networks, ENAS uses REINFORCE with a moving average baseline to reduce variance.

```python
<micro_controller.py>

for all index:
    curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=index)
    log_prob += curr_log_prob
    curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.nn.softmax(logits)))
    entropy += curr_ent

for all op_id:
    curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=op_id)
    log_prob += curr_log_prob
    curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.nn.softmax(logits)))
    entropy += curr_ent

arc_seq_1, entropy_1, log_prob_1, c, h = self._build_sampler(use_bias=True) # for convolution cell
arc_seq_2, entropy_2, log_prob_2, _, _ = self._build_sampler(prev_c=c, prev_h=h) # for reduction cell 
self.sample_entropy = entropy_1 + entropy_2
self.sample_log_prob = log_prob_1 + log_prob_2    
```

```python
self.valid_acc = (tf.to_float(child_model.valid_shuffle_acc) /
                      tf.to_float(child_model.batch_size))
    self.reward = self.valid_acc 

    if self.entropy_weight is not None:
      self.reward += self.entropy_weight * self.sample_entropy

    self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    baseline_update = tf.assign_sub(
      self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

    with tf.control_dependencies([baseline_update]):
      self.reward = tf.identity(self.reward)

    self.loss = self.sample_log_prob * (self.reward - self.baseline)
```

### 3. Child Network 

(1) Schematic of Child Network

<br/>![사진5](https://github.com/MINGUKKANG/ENAS-Tensorflow/blob/master/images/Child_Network.png)


## References
**Paper: https://arxiv.org/abs/1802.03268**

**Autors' implementation: https://github.com/melodyguan/enas**

**Data Pipeline: https://github.com/MINGUKKANG/MNIST-Tensorflow-Code**

## License
All rights related to this code are reserved to the author of ENAS

(Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean)
