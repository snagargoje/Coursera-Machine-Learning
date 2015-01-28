function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

return;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% 
% bestc=0;
% bestsig=0;
% besterr=1;
% for cc = [0 0.01 0.03 0.1 0.3 1 3 10 30 90 270 810 1000]
%     for sig = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
%         
%         cc
%         sig
%         model = svmTrain(X, y, cc, @(x1, x2) gaussianKernel(x1, x2, sig));
%         ycap = svmPredict(model,Xval);
%         
%         err = mean(double(ycap ~= yval))
%         
%         if besterr > err
%             besterr = err;
%             bestc = cc;
%             bestsig = sig;
%         end
%         
%     end 
% end
% besterr
% C=bestc
% sigma=bestsig
% 
% 

% =========================================================================

end
