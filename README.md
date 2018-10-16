# Collective Knowledge repository for PyTorch

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Introduction

This repository provides high-level, portable and customizable workflows
for [PyTorch](http://pytorch.org) 
as a part of our long-term community initiative
to [unify and automate AI](http://cKnowledge.org/ai) 
using [Collective Knowledge Framework (CK)](http://github.com/ctuning/ck/wiki).

## Coordination of development

* [cTuning Foundation](http://cTuning.org)
* [dividiti](http://dividiti.com)

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
