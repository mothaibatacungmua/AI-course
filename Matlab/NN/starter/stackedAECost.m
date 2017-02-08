function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% feedfoward pass
z1 = stack{1}.w * data + repmat(stack{1}.b,1,M);
h1 = sigmoid(z1);
z2 = stack{2}.w * h1 + repmat(stack{2}.b,1,M);
h2 = sigmoid(z2);
z3 = softmaxTheta * h2;
z3 = z3 - repmat(max(z3), numClasses, 1);
exp_ = exp(z3);
sum_exp_ = sum(exp_,1);
softmax = exp_ ./ repmat(sum_exp_, numClasses, 1);
tiny = exp(-30);
cost = -sum(diag(log(softmax + tiny) * groundTruth'))/M + ... 
       (lambda/2) * sum(sum(softmaxTheta.^2)) + ...
       (lambda/2) * sum(sum(stack{2}.w.^2)) + ...
       (lambda/2) * sum(sum(stack{1}.w.^2));

%backward pass
error_deriv = -(groundTruth - softmax)/M;
softmaxThetaGrad = error_deriv * h2' + lambda * softmaxTheta;

net2_deriv = (softmaxTheta' * error_deriv) .* h2 .* (1 - h2);
stackgrad{2}.w = net2_deriv * h1' + lambda * stack{2}.w;
stackgrad{2}.b = sum(net2_deriv,2);

net1_deriv = stack{2}.w' * net2_deriv .* h1 .* (1 - h1);
stackgrad{1}.w = net1_deriv * data' + lambda * stack{1}.w;
stackgrad{1}.b = sum(net1_deriv,2);







% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
