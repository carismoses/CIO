# CIO

This package implements Contact Invariant Optimization by Mordatch, et al.

## Install

This package has been tested with the following installation methods on Mac OS X 10.13.6

### Local Installation
If you plan on frequently interacting with the source code, or you would like to follow along with a notebook tutorial, perform a local installation with these steps
```
git clone git@github.com:carismoses/CIO.git
cd CIO
python3 setup.py develop
```
Installing the package this ways allows you to later uninstall with the following command
```
python3 setup.py develop --uninstall
```
(There is currently no equivalent for uninstalling if you use ```pip install -e .``` for the local installation)

### Git Installation
If you just want to be able to import this package and don't plan on changing anything, you can perform the following installation
```
pip install git+git://github.com/carismoses/CIO.git
```

## Test
If you followed the local installation, you can test a minimal example with the following commands
```
cd example
python3 minimal.py
```
WARNING: This will take a while to run!

## Tutorial
If you followed the local installation, you can follow a tutorial on how to use the package by launching a jupyter notebook with the following command
```
jupyter notebook notebook_tutorial.ipynb
```

## Tips
If you followed the local installation, running the minimal example with the following command will launch pdb (The Python Debugger)
```
python3 minimal.py --debug
```
To create a breakpoint, the paths have to start from the root of the source code. For example to set a breakpoint T line 80 in CIO/cio/world.py use the following command within pdb
```
break cio\world.py:80
```
