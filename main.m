% Test script to test a N layer(s) Neural Network

	%% Initialization
	clear; close all; clc;
	
	%% Setup the hyperparameters
	learning_rate = 0.3;
	weight_decay = 0.5;
	epoch = 100;
	
	%% Load the training data
	fprintf('\nLoading the training data...\n');
	X = loadMNISTImages("MNIST_data/train-images.idx3-ubyte")';
	y = loadMNISTLabels("MNIST_data/train-labels.idx1-ubyte");
	y(y == 0) = 10;
	m = size(X, 1);
	
	%% Load the validation data
	fprintf("Loading the validation data...\n");
	X_val = loadMNISTImages("MNIST_data/t10k-images.idx3-ubyte")';
	y_val = loadMNISTLabels("MNIST_data/t10k-labels.idx1-ubyte");
	y_val(y_val == 0) = 10;
	m_val = size(X_val, 1);
	
	%% Setup the neural network
	input_layer_size = size(X, 2);
	hidden_layers_size = 25;
	output_layer_size = 10;
	layers_size = [input_layer_size hidden_layers_size output_layer_size];
	
	%% Create the answer matrix
	fprintf("Creating the matrix containing the answers...\n");
	answer = zeros(m, output_layer_size);
	for i = 1:m
		answer(i, y(i)) = 1;
	end
	y = answer;
	clear answer;
	
	%% Create and initialize the weight matrices
	nn_params = XavierWeightInitialization(layers_size);
	
	%% Training Neural Network
	fprintf("\nTraining the Neural Network...\n");
	
	% Edit the options of the optimization function
	options = optimset("MaxIter", epoch);
	nnCostFunc = @(p) costFunction(p, layers_size, X, y, weight_decay);
	% Call the optimization function
	[nn_params, cost] = fmincg(nnCostFunc, nn_params, options);
	
	% Plot the cost
	plot(cost, '-b', 'LineWidth', 2);
	xlabel("Number of iteration");
	ylabel("Cost J");
	title("Training loss");
	
	% Compute the accuracy
	predictions = predict(nn_params, layers_size, X_val, true);
	accuracy = mean(double(y_val == predictions)) * 100;
	fprintf("Accuracy -> %f\n", accuracy);
	
%  Script end