function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
z = theta * data;
z = z - repmat(max(z), numClasses, 1);
exp_z = exp(z);
sum_exp_z = sum(exp_z,1);
softmax = exp_z ./ repmat(sum_exp_z, numClasses, 1);
tiny = exp(-30);
cost = -sum(diag(log(softmax + tiny) * groundTruth'))/numCases + (lambda/2) * sum(sum(theta.^2));
thetagrad = -(groundTruth - softmax) * data'/numCases + lambda * theta;









% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

