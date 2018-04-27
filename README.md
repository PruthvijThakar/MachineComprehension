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

## Tasks
-----

The tasks in ``babi/tasks`` correspond to those from the original dataset as
follows:

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |


|   Task |                                        | Class name |
| -------- |----------- |
| ---                                              | ---            |
| 1  Basic factoid QA with single supporting fact | WhereIsActor |
| 2  Factoid QA with two supporting facts         | WhereIsObject |
| 3  Factoid QA with three supporting facts       | WhereWasObject |
| 4  Two argument relations: subject vs. object   | IsDir |
| 5  Three argument relations                     | WhoWhatGave |
| 6  Yes/No questions                             | IsActorThere|
| 7  Counting                                     | Counting |
| 8  Lists/Sets                                   | Listing |
| 9  Simple Negation                              | Negatio |
| 10  Indefinite Knowledge                         | Indefinite |
| 11  Basic coreference                            | BasicCoreference |
| 12  Conjunction                                  | Conjunction |
| 13  Compound coreference                         | CompoundCoreference |
| 14  Time manipulation                            | Time |
| 15  Basic deduction                              | Deduction |
| 16  Basic induction                              | Induction |
| 17  Positional reasoning                         | PositionalReasoning |
| 18  Reasoning about size                         | Size |
| 19  Path finding                                 | PathFinding |
| 20  Reasoning about agent's motivation           | Motivations |


## OVERVIEW

*	Neural Network model with external memory.
*	Reads the memory with soft attention.
*	It accesses memory multiple times; each step being called a hop.
*	Uses back propagation to update the model.


## Requirements
* Python 3 and above
* Numpy, Flask, TensorFlow (only for web-based demo) can be installed via pip:

* [bAbI dataset](http://fb.ai/babi) should be downloaded to `data/tasks_1-20_v1-2`: 

```
$ wget -qO- http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz | tar xvz -C data
```
```
docker pull tushargl016/cognitivefinalproject
docker run -p 80:8000 -d -ti tushargl016/cognitivefinalproject
```

### References
* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, 
  "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)",
  *arXiv:1503.08895 [cs.NE]*.

### Link to Web: http://ec2-54-161-78-119.compute-1.amazonaws.com/


