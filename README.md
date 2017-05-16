# MNIST Classification with a N-layer Neural Network

Train a N-layer Neural Network on MNIST dataset using Octave. Mainly inspired by Andrew Ng's Machine Learning course.

---

## Prerequisities
Please download first the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) on Yann Lecun's website.
Then, create a new folder and name it **MNIST_data**
Finally, extract them in that folder. You should have the following file tree:
- MNIST_data/
	- t10k-images.idx3-ubyte
	- t10k-labels.idx1-ubyte
	- train-images.idx3-ubyte
	- train-labels.idx1-ubyte
	
## Running the tests
In the **main.m** script, you can choose the following hyperparameters:
- Learning rate
- Weight decay
- Number of layer
- Number of neauron in each layer