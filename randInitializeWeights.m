function nn_params = randInitializeWeights(layers_size, epsilon_init)
	% RANDINITIALIZEWEIGHTS Randomly initialize the weights of the neural network.
	% This way, we break the symmetry while training the neural network.
	
	nn_params = [];
	for i = 1:(length(layers_size) - 1)
		nn_params = [zeros(layers_size(i + 1) * (layers_size(i) + 1), 1) ; nn_params];
	end

	nn_params = randn(length(nn_params), 1) * (2 * epsilon_init) - epsilon_init;
end