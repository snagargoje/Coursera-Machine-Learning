function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% size(X)
% size(theta)
% size(y)

% 12 x 2 * 2 x 1 = 12 x 1
part1 = ((X*theta - y)' * (X*theta - y) ) / (2 * m);


part2 = ( (theta(2:end)' * theta(2:end)) * lambda ) / (2 * m);

J = part1 + part2;

n=size(X,2);
% 12 x 1 * 1 x 2 
part1 = ( sum( X .* ( (X*theta - y) * ones(1,n) ) ) / m )';

part2 = theta * lambda / m;
part2(1)=0;

grad = part1 + part2;











% =========================================================================

grad = grad(:);

end
