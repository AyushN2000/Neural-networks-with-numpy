# Neural Networks using numpy
This code let's you have all the flexibility you would want in terms of model architechture in fully connected layers, and the input data pipeline.

## Requirements
* numpy(v1.17) : It is used for all the matrix operations.
* tqdm(v4.45) : It is used to display the progress bar while training occurs
* matplotlib(v3.2.1) : This module comes in handy with visualizing training metrics and gives us an insight about the performance of the model
* jupyter(v6.0.3) : This module provides a web interface to run python scripts in form of notebooks.
* tensorflow(v2.0.0a0) : To load MNIST dataset of handwritten digits.
If you have your own dataset, you can omit this module. 

## Usage-Running the notebook:
1. Clone this repository, open terminal or CMD prompt at the cloned directory's path. 
1. Check for availability for the python modules specidied above in **Requirements**.
1. If modules aren't installed, run the following command at terminal or CMD prompt.

   ` $ pip install <module_name> `
 1. After installation of **Requirements** Run the following command while in terminal and click on the button  named **Run** , after the jupyter notebook opens up.

    ` $ jupyter notebook Neural_network.ipynb `
 1. If you want to train a custom network, (and/or) add dense layers with different parameters as in this test notebook,you'll just have to add `network.append(Layers().Dense(<input nodes>,<output nodes>)` and `network.append(activations().ReLU())` to add the layers.
 
 While adding layers or changing the number input/output nodes, remember to keep the number of output nodes of the previous layer equal to the number of input nodes of the current layer.
 
 ## Future scope:
 The motivation to take up this project was to make a clone of Tensorflow's keras API which is a high-level API used for building and training Deep Learning models.
 
 The codebase developed is far from ideal, and has lots of scope for improvement in performance and overall reduction of training time. In the upcoming commits, support for different activation functions like **sigmoid()** and **tanh()** can be added.
 
 Like the Dense class, a class to represent Convolutional Neural Network(CNN) layers can be included. This opens up the scope of use cases for this codebase to a wide spectrum of Image classifications and other Computer Vision problems.
