function s = softmax(z)
	% SOFTMAX Computes the softmax function in a numerically stable way.
	%	s = softmax(z) Applies the softmax function to the input array.
	%
	%	Usually, the softmax function is the following:
	%		s = exp(vector(i))/sum(exp(vector))
	%
	%	What we do here is exactly equivalent, but this is more numerically 
	%	stable.
	%	"Numerically stable" means that this way, there will never be really 
	%	big numbers involved.
	
	dim = 2;
    % tmp = ones(1, ndims(z));
    % tmp(dim) = size(z, dim);
    % maxz = max(z, [], dim);
    % expz = exp(z - repmat(maxz, tmp));
    % s = expz ./ repmat(sum(expz, dim), tmp);
	
	s = exp(z - max(z, dim));
	s = s / sum(s, dim);
end