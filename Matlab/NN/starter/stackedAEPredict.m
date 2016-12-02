function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

M = size(data, 2);
z1 = stack{1}.w * data + repmat(stack{1}.b,1,M);
h1 = sigmoid(z1);
z2 = stack{2}.w * h1 + repmat(stack{2}.b,1,M);
h2 = sigmoid(z2);
z3 = softmaxTheta * h2;
z3 = z3 - repmat(max(z3), numClasses, 1);
exp_ = exp(z3);
sum_exp_ = sum(exp_,1);
softmax = exp_ ./ repmat(sum_exp_, numClasses, 1);


[value,pred] = max(softmax);







% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
