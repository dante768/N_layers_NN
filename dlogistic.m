function ret = dlogistic(input)
	% DLOGISTIC Returns the derivative of the logistic function given the input
	ret = logistic(input) .* (1 - logistic(input));
end