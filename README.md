# A-test-code-for-federated-learning-
Test code of a federated learning model with two remote workers. 
Modified based on a PySyft official site tutorial.

Includes:
server code, 
client code, 
dataset, 
Some instructions on Pytorch and Pysyft installations, 
documentations on websocket communication

There maybe some issues with implemenation:
1.	Location of python packages installed at centralized server: directly using pip install in conda environment does not actually install the packages to the currently activated conda environment. Instead, packages are installed to the system environment, which is also a different environment with conda base environment. Therefore, to install packages from PyPI channel, command python -m pip install should be used.
2.	PyTorch installation on Raspberry PI: most existing PyTorch wheels run into compatibility issues when deployed on raspberry Pi. Newer versions of GNU library are required which is an operating system level error that cannot be satisfied by current Raspbian Buster release. Lower version of PyTorch is installed from an appropriate wheel file.
