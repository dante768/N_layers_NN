function checkNNGradients(lambda)
	% CHECKNNGRADIENTS Creates a small neural network to check the 
	% backpropagation gradients.
	%	CHECKNNGRADIENTS(lambda) Creates a small neural network to check the 
	%	backpropagation gradients. It will output the analytical gradients 
	%	produced by the backpropagation code and the numerical gradients.
	%	These two gradients computations should result in very similar values.
	
	if ~exist('lambda', 'var') || isempty(lambda)
		lambda = 0;
	end
	
	%% Setup the neural network
	input_layer_size = 3;
	hidden_layers_size = 5;
	output_layer_size = 3;
	layers_size = [input_layer_size hidden_layers_size output_layer_size];
	m = 5;
	
	% Generates some "random" test data
	Theta1 = debugInitializeWeights(hidden_layers_size, input_layer_size);
	Theta2 = debugInitializeWeights(output_layer_size, hidden_layers_size);
	% Reusing debugInitializeWeights to generate X
	X = debugInitializeWeights(m, input_layer_size - 1);
	y = 1 + mod(1:m, output_layer_size)';
	
	%% Create the answer matrix
	answer = [];
	for i = 1:m
		tmp = zeros(1, output_layer_size);
		tmp(y(i)) = 1;
		answer = [answer; tmp];
	end
	y = answer;
	clear answer;
	
	% Unroll parameters
	nn_params = [Theta1(:) ; Theta2(:)];
	
	% Short hand for cost function
	costFunc = @(p) costFunction(p, layers_size, X, y, lambda);
	
	% Compute the numerical gradient
	numgrad = computeNumericalGradient(costFunc, nn_params);
	
	% Compute analytical gradient
	[cost, grad] = costFunction(nn_params, layers_size, X, y, lambda);
	
	% Visually examine the two gradients computations. The two colums you get 
	% should be very similar
	fprintf(["The two colums below you get should be very similar.\n" ...
			 "(Left - Numerical Gradient, Right - Analytical Gradient)\n\n"]);
	disp([numgrad grad]);
	
	% Evaluate the norm of the difference between the two solutions.
	% Assuming EPSILON = 0.0001 in computeNumericalGradient.m, then diff should 
	% be less than 1e-9
	diff = norm(numgrad - grad) / norm(numgrad + grad);
	fprintf("Relative Difference: %g\n", diff);
end