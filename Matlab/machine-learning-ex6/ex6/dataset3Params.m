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
sigma = 0.3;

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

C_vec = 1:1:10;
sigma_vec = 0.1:0.1:1;
min_index = ones(1, 2);
pred_min = 1.0;

for i=1:size(C_vec,2)
    for j=1:size(sigma_vec,2)
        model = svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        predictions = svmPredict(model, Xval);
        val_error = mean(double(predictions ~= yval));
        if pred_min > val_error
            pred_min = val_error;
            min_index(1) = i;
            min_index(2) = j;
        end
    end
end

C = C_vec(min_index(1));
sigma = sigma_vec(min_index(2));




% =========================================================================

end
