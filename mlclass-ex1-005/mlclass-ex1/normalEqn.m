function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
n=size(X,1); % 97
%X = [ones(n, 1), X(:,1:end)]; % Add a column of ones to x
theta = pinv(X'*X)*X'*y;


% -------------------------------------------------------------


% ============================================================

end
