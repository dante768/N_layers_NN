function [J grad] = costFunction( nn_params, ...
							layers_size, ...
							X, y, lambda)
	% COSTFUNCTION Implements the neural network cost function for a N layer(s)
	% neural network which performs classification
	%	[J grad] = COSTFUNCTION(nn_params, layers_size, X, y, lambda, momentum) 
	%	computes the cost and the gradient of the neural network. 
	%
	%	- The parameters for the neural network are "unrolled" into the vector nn_params. 
	%	- layers_size is a vector of integer which contains the number of elements of 
	%	each layer.
	%	- X is a matrix of size <number of examples> by <number of features>
	%	- y is a matrix of size <number of examples> by <number of classes
	%
	%	The returned parameter grad is a "unrolled" vector of the partial derivatives 
	%	of the neural network.
	
	if ~exist("lambda", "var") || isempty(lambda)
		lambda = 0;
	end
	
	layer_nb = length(layers_size);
	if(layer_nb < 3)
		error('At least 3 layers are necessary.');
	end
	
	% Reshape nn_params back into the parameter 3D matrix Thetas, the weight matrices 
	% for our N layer neural networks
	Thetas = [];
	layers_size_tmp = [1 layers_size];
	
	idx_1 = layers_size_tmp(1);
	idx_2 = 0;
	for i = 1:(layer_nb - 1)
		idx_2 = idx_2 + layers_size_tmp(i+2) * (layers_size_tmp(i+1) + 1);
		
		Thetas(i).mat = reshape(nn_params(idx_1:idx_2), layers_size_tmp(i+2), (layers_size_tmp(i+1) + 1));
								
		idx_1 = idx_2 + 1;
	end
	
	% Variables to return
	J = 0; regularization_term = 0; grad = [];
	% Useful variables
	layer_activities = [];
	before_layer_activities = [];
	m = size(X, 1);
	
	% Part 1
	% - Feedforward phase
	for i = 1:layer_nb
		before_layer_activities(i).mat = zeros(m, layers_size(i));
		layer_activities(i).mat = zeros(m, layers_size(i));
	end
	
	for i = 1:(layer_nb-1)
		layer_activities(i).mat = [ones(size(layer_activities(i).mat, 1), 1), layer_activities(i).mat];
		before_layer_activities(i + 1).mat = layer_activities(i).mat * (Thetas(i).mat)';
		layer_activities(i + 1).mat = logistic(before_layer_activities(i + 1).mat);
	end
	
	% - Unregularized cost function
	for i = 1:m
		J = J + ((y(i, :) * log(layer_activities(end).mat(i, :))') + ((1 - y(i, :)) * log(1 - layer_activities(end).mat(i, :))'));
	end
	J = -(J/m);
	
	% - Regularized cost function
	for i = 1:size(Thetas, 2)
		tmp_theta = Thetas(i).mat;
		for j = 2:size(tmp_theta, 2)
			regularization_term = regularization_term + (tmp_theta(:, j)' * tmp_theta(:, j));
		end
	end
	J = J + ((lambda/(2*m)) * regularization_term);
	
	% Part 2
	% - Backpropagation
	Thetas_grad = [];
	deltas = [];
	grad = [];
	
	for i = 1:size(Thetas, 2)
		Thetas_grad(i).mat = zeros(size(Thetas(i).mat));
	end
	
	for i = 1:layer_nb
		deltas(i).mat = zeros(m, layers_size(i));
	end
	
	deltas(end).mat = layer_activities(end).mat - y;
	
	for i = (size(deltas, 2)-1):-1:1
		for j = 1:m
			% Unregularized gradients
			deltas(i).mat(j,:) = Thetas(i).mat(:,2:end)' * deltas(i + 1).mat(j,:)' .* dlogistic(before_layer_activities(i).mat(j,:))';
			Thetas_grad(i).mat = Thetas_grad(i).mat + ((layer_activities(i).mat(j,:)' * deltas(i + 1).mat(j,:)))';
		end
		Thetas_grad(i).mat = Thetas_grad(i).mat/m;
		% Regularized gradients
		Thetas_grad(i).mat = Thetas_grad(i).mat + ((lambda/m) * [zeros(size(Thetas(i).mat, 1), 1), Thetas(i).mat(:,2:end)]);
		grad = [Thetas_grad(i).mat(:) ; grad];
	end	
end