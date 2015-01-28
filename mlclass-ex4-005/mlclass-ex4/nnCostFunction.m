function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% for i=1:m
%   
% X 5000x400
% Theta1 25x401
% Theta2 10x26

    X=[ones(m,1) X];


    zz2 = X*Theta1';
%     sizeof_zz1=size(zz2)
%     size(zz2,2)
    for i=1:size(zz2,2)
        zz2(:,i)=ones(size(zz2(:,i))) ./ (ones(size(zz2(:,i))) + exp(-zz2(:,i)));
    end
    zz2= [ones(size(zz2,1),1) zz2];
    zz3= zz2*Theta2';
%     sizeof_zz3=size(zz3)
    for i=1:size(zz3,2)
        zz3(:,i)=ones(size(zz3(:,i))) ./ (ones(size(zz3(:,i))) + exp(-zz3(:,i)));
    end
%     sizeof_zz3=size(zz3)
    
    tmp=zeros(m,num_labels);
    for i=1:m
       tmp(i,y(i))=1; 
    end
    
    
    logans3 = log(zz3);
    log_one_minus_log_ans3=log(ones(size(zz3))-zz3);
    
    part1 = -tmp.*logans3;
    part2= -(ones(size(tmp)) - tmp) .* log_one_minus_log_ans3;
    
    part3 = part1+part2;
    finalans1=sum(part3,2);
    finalans2=sum(finalans1,1);

    opt_cost=finalans2/m;
    
    t1=Theta1(:,2:end).*Theta1(:,2:end);
    t1=sum(t1,2);
    t1=sum(t1,1);
    
    t2=Theta2(:,2:end).*Theta2(:,2:end);
    t2=sum(t2,2);
    t2=sum(t2,1);
    
    reg_cost=(lambda/(2*m)) * (t1+t2);
    
    J=opt_cost + reg_cost;
    
    
    del3=[];
    del2=[];
    grad1=zeros(size(Theta1));
    grad2=zeros(size(Theta2));
    
    for i=1:m
       xi = X(i,:);
       
        zz2 = xi*Theta1';
%         sizeof_zz1=size(zz2)
%         size(zz2,2)
        aa2=ones(size(zz2)) ./ (ones(size(zz2)) + exp(-zz2));
        aa2= [1 aa2];
        zz3= aa2*Theta2';
%         sizeof_zz3=size(zz3)
        aa3=ones(size(zz3)) ./ (ones(size(zz3)) + exp(-zz3));
%         sizeof_aa3=size(aa3)
        
        hx=aa3;
        
        yk=zeros(size(aa3));
        yk(y(i))=1;
        del3 = aa3-yk;

        del3=del3';
%         sizeof1=size(Theta2)
%         sizeof2=size(del3)
%         sizeof3=size(zz2)
%         sizeof4=size(Theta2'*del3)
        
        
        del2=Theta2'*del3 .* sigmoidGradient([1 zz2]');
        
        del2 = del2(2:end);
        
%         size(del3)
%         size(aa2)
        grad2 = grad2 + del3*aa2;
        grad1 = grad1 + del2*xi;
        
    end
    
    
    
    
%     [ans4,I]=max(ans3,[], 2);

% end

% n=size(X,1); % 97
% m=size(X,2); % 2
% o=ones(1,n); % 1xn
% ans = (theta*o)'; %2x1 * 1xn = 2xn
% ans = ans .* X; %nx2
% a=sum(ans,2); %nx1
% 
% g = ones(size(a)) ./ (ones(size(a)) + exp(-a));
% 
% part1 = - y .* log(g);
% part2 = -(ones(size(y)) - y) .* log(ones(size(y)) - g);
% part3 = part1 + part2;
% final = sum(part3);
% final = final/n;
% 
% temptheta = theta;
% temptheta(1)=0;
% final2 = temptheta' * temptheta;
% final2 = final2 * lambda;
% final2 = final2 /(2*n);
% 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad1=grad1/m;
grad2=grad2/m;

part1 = Theta1 * lambda / m;
part2 = Theta2 * lambda / m;

grad1(2:end,2:end) =  grad1(2:end,2:end) + part1(2:end,2:end) ;
grad2(2:end,2:end) =  grad2(2:end,2:end) + part2(2:end,2:end) ;



grad = [grad1(:) ; grad2(:)];
% size_of_grad=size(grad)

end
