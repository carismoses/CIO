# CIO

This package implements Contact Invariant Optimization by Mordatch, et al.

## Install

This package has been tested with the following installation methods on Mac OS X 10.13.6

### Local Installation
If you plan on frequently developing the package, perform a local installation with these steps
```
git clone git@github.com:carismoses/CIO.git
cd CIO
python3 setup.py install
```
installing the package this ways allows you to later uninstall with the following command
```
python3 setup.py uninstall
```
There is currently no equivalent for uninstalling if you use ```pip install -e .``` for the local installation

### Git Installation
If you just want to be able to import this package and don't plan on changing anything, you can perform the following installation.
```
pip install git+git://github.com/carismoses/CIO.git
```

## Test
If you have cloned the package, you can test a minimal example with the following commands
```
cd examples
python3 minimal.py
```

## Tutorial
If you have cloned the package, you can follow a tutorial on how to use the package by launching a jupyter notebook with the following command
```
jupyter notebook
```
Then, open the ```CIO_notebook.ipynb``` file.
