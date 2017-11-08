from network_layers import *
from transfer_functions import * 
from neural_net import * 
from genetic_algorithm import * 

import matplotlib.pyplot as plt
import numpy as np
import data_loader
import os, sys 

###
#   Global constants, I/O paths
###

SIN_TRAIN = os.path.join('data','sine_train.txt')
SIN_TEST = os.path.join('data','sine_test.txt')

RASTRIGIN_TRAIN = os.path.join('data','rastrigin_train.txt')
RASTRIGIN_TEST = os.path.join('data','rastrigin_test.txt')

ROSENBROCK_TRAIN = os.path.join('data','rosenbrock_train.txt')
ROSENBROCK_TEST = os.path.join('data','rosenbrock_test.txt')

ML_TRAIN = os.path.join('data', 'train_hrvoje.txt')
ML_LOCAL_TEST = os.path.join('data', 'test_hrvoje.txt')
ML_TEST = os.path.join('data', 'test_hrvoje_sve.txt')

if __name__ == '__main__':
	# set the random seed for reproducibility of results
	# setting the random seed forces the same results of randoming each
	# time you start the program - that way you can demonstrate your results
	np.random.seed(11071998)


	# Load the train / test data
	# X is the input matrix, y is the target vector
	# X can be a vector (and will be, in the first assignment) as well 

	"""
		To change the function being approximated, just change the paths 
		to the dataset in the arguments of the data loader.s
	"""
	X_train, y_train = data_loader.loadFrom(ML_TRAIN)
	X_local_test, y_local_test = data_loader.loadFrom(ML_LOCAL_TEST)
	X_test, y_test = data_loader.loadFrom(ML_TEST)

	X_train = X_train[:100]
	y_train = y_train[:100]

	# for check, print out the shapes of the input variables
	# the first dimension is the number of input samples, the second dimension
	# is the number of variables 

	print("Train data shapes: ", X_train.shape, y_train.shape)
	print("Test data shapes: ", X_test.shape, y_test.shape)

	# Insert
	#sys.exit()

	# The dimensionality of the input layer of the network is the second
	# dimension of the shape 

	if len(X_train.shape) > 1:
		input_size = X_train.shape[1]
	else: 
		input_size = 1

	# the size of the output layer
	output_size = 1

	NN = NeuralNetwork()

	#  Define the layers of your
	#        neural networks
	#############################
	#       YOUR CODE HERE      #
	#############################

	NN.addLayer(Neuron(input_size, input_size))
	#NN.addLayer(Neuron(2 * input_size, 2 * input_size))
	NN.addLayer(LinearLayer(input_size ,output_size))


	####################
	#  YOUR CODE ENDS  #
	####################

	def errorClosure(w):
		"""
			A closure is a variable that stores a function along with the environment.
			The environment, in this case are the variables x, y as well as the NN
			object representing a neural net. We store them by defining a method inside
			a method where those values have been initialized. This is a "hacky" way of 
			enforcing the genetic algorithm to work in a generalized manner. This way,
			the genetic algorithm can be applied to any problem that optimizes an error 
			(in this case, this function) by updating a vector of values (in this case,
			defined only by the initial size of the vector). 

			In plain - the genetic algorithm doesn't know that the neural network exists,
			and the neural network doesn't know that the genetic algorithm exists. 
		"""
		# Set the weights to the pre-defined network
		NN.setWeights(w)
		# Do a forward pass of the etwork and evaluate the error according to the
		# oracle (y)
		return NN.forwardStep(X_train, y_train)

	# Check the constructor (__init__) of the GeneticAlgorithm for further instructions
	# on what the parameters are. Feel free to change / adapt any parameters. The defaults
	# are as follows 


	#######################################
	#    MODIFY CODE AT WILL FROM HERE    #
	#######################################

	elitism = 4 # Keep this many of top units in each iteration
	populationSize = 15 # The number of chromosomes
	mutationProbability  = .12 # Probability of mutation
	mutationScale = .11 # Standard deviation of the gaussian noise
	numIterations = 1000 # Number of iterations to run the genetic algorithm for
	errorTreshold = 1e-6 # Lower threshold for the error while optimizing

	GA = GeneticAlgorithm(NN.size(), errorClosure,
		elitism = elitism,
		populationSize = populationSize,
		mutationProbability = mutationProbability,
		mutationScale = mutationScale, 
		numIterations = numIterations, 
		errorTreshold = errorTreshold)

	print_every = 100 # Print the output every this many iterations

	# emulated do-while loop
	done = False
	while not done: 
		done, iteration, best = GA.step()

		if iteration % print_every == 0: 
			print("Error at iteration %d = %f" % (iteration, errorClosure(best)))

	print("Training done, running on local test set")
	NN.setWeights(best)
	print("Error on local test set: ", NN.forwardStep(X_local_test, y_local_test))

	test_output	= np.array(NN.outputs(X_test)).flatten()
	test_output = list(map(lambda d: -1 if d <= 0 else 1, test_output))
	
	data_loader.writeTo(test_output)
	
