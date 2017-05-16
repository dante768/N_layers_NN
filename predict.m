function p = predict(nn_params, layers_size, X, return_index)
	% PREDICT Predict the label of an input given a trained neural network
	%	p = predict(nn_params, layers_size, X) output the predicted label of 
	%	X gievn the trained weights of a neural network
	
	% Reshape nn_params back into the parameter 3D matrix Thetas, the weight matrices 
	% for our N layer neural networks
	
	if ~exist("return_index", "var") || isempty(return_index)
		return_index = false;
	end
	
	% Useful variables
	Thetas = [];
	layers_size_tmp = [1 layers_size];
	layer_nb = length(layers_size);
	
	idx_1 = layers_size_tmp(1);
	idx_2 = 0;
	for i = 1:(layer_nb - 1)
		idx_2 = idx_2 + layers_size_tmp(i+2) * (layers_size_tmp(i+1) + 1);
		
		Thetas(i).mat = reshape(nn_params(idx_1:idx_2), layers_size_tmp(i+2), (layers_size_tmp(i+1) + 1));
								
		idx_1 = idx_2 + 1;
	end
	
	% Useful variables
	before_layer_activities = [];
	layer_activities = [];
	m = size(X, 1);
	
	for i = 1:layer_nb
		before_layer_activities(i).mat = zeros(m, layers_size(i));
		layer_activities(i).mat = zeros(m, layers_size(i));
	end
	layer_activities(1).mat = X;
	
	for i = 1:(layer_nb-1)
		layer_activities(i).mat = [ones(size(layer_activities(i).mat, 1), 1), layer_activities(i).mat];
		before_layer_activities(i + 1).mat = layer_activities(i).mat * (Thetas(i).mat)';
		layer_activities(i + 1).mat = logistic(before_layer_activities(i + 1).mat);
	end
	
	for i = 1:m
		% Compute the softmax function to represent a probability distribution across all discrete alternatives
		layer_activities(end).mat(i,:) = softmax(layer_activities(end).mat(i,:));
	end
	
	% Get the index of the maximum probability
	[dummy, idx] = max(layer_activities(end).mat, [], 2);
	
	if return_index == false
		% Useful variables
		final_predictions = zeros(size(layer_activities(end).mat));
		n_case = size(layer_activities(end).mat, 2);
		
		for i = 1:m
			final_predictions(i, idx(i)) = 1;
		end
		p = final_predictions;
	else
		p = idx;
	end
end