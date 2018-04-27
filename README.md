# MachineComprehension

## Problem Statement

Example: Task with single supporting fact

#### Comprehension:
John is in the kitchen.
Bob is in the gardeb.

#### Query:
Where is John? 

#### Answer:
kitchen (one with max. probability)

## OVERVIEW

●	Neural Network model with external memory.
●	Reads the memory with soft attention.
●	It accesses memory multiple times; each step being called a hop.
●	Uses back propagation to update the model.



## Requirements
* Python 3 and above
* Numpy, Flask, TensorFlow (only for web-based demo) can be installed via pip:

* [bAbI dataset](http://fb.ai/babi) should be downloaded to `data/tasks_1-20_v1-2`: 

```
$ wget -qO- http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz | tar xvz -C data
```

### References
* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, 
  "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)",
  *arXiv:1503.08895 [cs.NE]*.

### Link to Web: http://ec2-54-161-78-119.compute-1.amazonaws.com/


