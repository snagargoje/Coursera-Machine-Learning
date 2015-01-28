function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%


n=size(X,1); % 97
m=size(X,2); % 2
o=ones(1,n); % 1xn
ans = (theta*o)'; %2x1 * 1xn = 2xn
ans = ans .* X; %nx2
a=sum(ans,2); %nx1

g = ones(size(a)) ./ (ones(size(a)) + exp(-a));

pos = find(g>=0.5);
neg = find(g<0.5);

ans = zeros(n,1);
ans(pos,:)=1;
p = ans;


% =========================================================================


end
