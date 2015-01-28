function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
size(X)
size(Theta1)
size(Theta2)
num_labels
size(p)
%         5000         400
%     25   401
%     10    26
%     10
%         5000           1

X=[ones(m,1) X];


ans1 = X*Theta1';
sizeof_ans1=size(ans1)
size(ans1,2)
for i=1:size(ans1,2)
    ans1(:,i)=ones(size(ans1(:,i))) ./ (ones(size(ans1(:,i))) + exp(-ans1(:,i)));
end
ans2= [ones(size(ans1,1),1) ans1];
ans3= ans2*Theta2';
sizeof_ans3=size(ans3)
for i=1:size(ans3,2)
    ans3(:,i)=ones(size(ans3(:,i))) ./ (ones(size(ans3(:,i))) + exp(-ans3(:,i)));
end
sizeof_ans3=size(ans3)

[ans4,I]=max(ans3,[], 2);

tens = find(I==num_labels);
% ans4(1:10)
% I(1:10)

size(ans4)
size(I)

% TODO why not apply below thing ... we are supposed to change labels from
% 10 to 0
% I(tens)=0;
p=I;

% 
% A=zeros(num_labels,n);
% ans=all_theta*X';
% ans = ans';
% for i=1:num_labels
%     g = ones(size(ans(:,i))) ./ (ones(size(ans(:,i))) + exp(-ans(:,i)));
%     A(i,:)=g;
% end
% 
% p=max(A, [], 1);
% 
% 
% tens = find(p==10);
% 
% p(tens)=0;
% p=p';

% =========================================================================
end



