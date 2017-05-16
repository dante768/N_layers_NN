function nn_params = XavierWeightInitialization(layers_size)
	% XAVIERWEIGHTINITIALIZATION Initialize the weights of the neural 
	% network by following the Xavier Initialization.
	%
	%	Pick the weight from a Gaussian distribution with zero mean and 
	%	a variance of 2 / (N_in + N_out) (where N_in specifies the number 
	%	of neurons of the input layer and N_out the number of neurons of the 
	%	output layer)
	
	nn_params = [];
	
	for i = 1:(length(layers_size) - 1)
		tmp = randn(layers_size(i + 1) * (layers_size(i) + 1), 1) * (2 / ((layers_size(i) + 1) + layers_size(i + 1)));
		nn_params = [nn_params ; tmp];
	end
	
end