% Test script to test a N layer(s) Neural Network

	%% Initialization
	clear; close all; clc;

	%% Setup the neural network
	input_layer_size = 400;
	hidden_layers_size = [200 200];
	output_layer_size = 10;
	layers_size = [input_layer_size hidden_layers_size output_layer_size];
	
	%% Setup the hyperparameters
	epsilon_init = 0.10;
	alpha = 0.01;
	lambda = 5;
	momentum = 0;
	epoch = 1e3;
	
	%% Load the training data
	fprintf('Loading the training data...\n');
	load("ex4data1.mat");
	m = size(X, 1);
	
	%% Create the answer matrix
	answer = [];
	for i = 1:m
		tmp = zeros(1, output_layer_size);
		tmp(y(i)) = 1;
		answer = [answer; tmp];
	end
	y = answer;
	clear answer;
	clear tmp;
	
	%% Create and initialize the weight matrices
	nn_params = randInitializeWeights(layers_size, epsilon_init);