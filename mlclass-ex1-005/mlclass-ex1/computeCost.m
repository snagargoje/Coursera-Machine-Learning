function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% n=size(X,1);
% o=ones(1,n);
% ans = (theta*o)';
% ans = ans .* X;
% a=sum(ans,2);
% b = a - y;

n=size(X,1); % 97
m=size(X,2); % 2
o=ones(1,n); % 1xn
ans = (theta*o)'; %2x1 * 1xn = 2xn
ans = ans .* X; %nx2
a=sum(ans,2); %nx1
b = a - y; %nx1

J = b'*b; %n
J=J/2;
J=J/n;
% =========================================================================

end
