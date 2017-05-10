function ret = logistic(input)
	% Applies the logistic function to each element of the input
	ret = 1 ./ (1 + exp(-input));
end