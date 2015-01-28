function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

n=size(X,1); % 97
m=size(X,2); % 2
o=ones(1,n); % 1xn
ans = (theta*o)'; %2x1 * 1xn = 2xn
ans = ans .* X; %nx2
a=sum(ans,2); %nx1

g = ones(size(a)) ./ (ones(size(a)) + exp(-a));

part1 = - y .* log(g);
part2 = -(ones(size(y)) - y) .* log(ones(size(y)) - g);
part3 = part1 + part2;
final = sum(part3);
final = final/n;
J=final;

inter = (g-y) * ones(1,m);
inter = (inter .* X );% .* eye(n,n);
inter = sum(inter,1);
inter = inter ./ n;

grad = inter';
% 
% size(theta)
% size(grad)

% =============================================================

end
