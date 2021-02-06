# Collective Knowledge repository for PyTorch

**All CK components can be found at [cKnowledge.io](https://cKnowledge.io) and in [one GitHub repository](https://github.com/ctuning/ai)!**

*This project is hosted by the [cTuning foundation](https://cTuning.org).*

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![automation](https://github.com/ctuning/ck-guide-images/blob/master/ck-artifact-automated-and-reusable.svg)](http://cTuning.org/ae)
[![workflow](https://github.com/ctuning/ck-guide-images/blob/master/ck-workflow.svg)](http://cKnowledge.org)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Introduction

This repository provides portable, customizable, and reproducible workflows, automation actions, and reusable artifacts
for [PyTorch](http://pytorch.org) in the [Collective Knowledge format (CK)](https://github.com/ctuning/ck).

## Minimal CK installation

The minimal installation requires:

* Python 2.7 or 3.3+ (limitation is mainly due to unitests)
* Git command line client.

### Linux/MacOS

You can install CK in your local user space as follows:

```
$ git clone http://github.com/ctuning/ck
$ export PATH=$PWD/ck/bin:$PATH
$ export PYTHONPATH=$PWD/ck:$PYTHONPATH
```

You can also install CK via PIP with sudo to avoid setting up environment variables yourself:

```
$ sudo pip install ck
```

### Windows

*We still need to provide proper support to build PyTorch via CK on Windows*

First you need to download and install a few dependencies from the following sites:

* Git: https://git-for-windows.github.io
* Minimal Python: https://www.python.org/downloads/windows

You can then install CK as follows:
```
 $ pip install ck
```

or


```
 $ git clone https://github.com/ctuning/ck.git ck-master
 $ set PATH={CURRENT PATH}\ck-master\bin;%PATH%
 $ set PYTHONPATH={CURRENT PATH}\ck-master;%PYTHONPATH%
```

## CK workflow installation for PyTorch 

### CPU

```
$ ck pull repo:ck-pytorch
$ ck install package --tags=lib,pytorch,vcpu
```

### GPU

```
$ ck pull repo:ck-pytorch
$ ck install package --tags=lib,pytorch,vcuda
```

## Checking classification example (and automatically installing available MXNet model(s) via CK)

```
$ ck install package --tags=lib,pytorch-vision
$ ck run program:pytorch
```

* Select 'classify-squeezenet-1.1'
* Select image to classify
* Observe result

## Next steps

We plan to add PyTorch to our ReQuEST tournament framework: http://cKnowledge.org/request

## Feedback

Get in touch with CK-AI community [here](https://github.com/ctuning/ck/wiki/Contacts). 
