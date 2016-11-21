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


NN_OUTPUT = size(Theta2, 1);
a_1 = [ones(m, 1) X];
z_2 = a_1*Theta1';
a_2 = [ones(size(z_2,1),1) sigmoid(z_2)];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
%convert y to matrix represention
Y = zeros(m, NN_OUTPUT);
for i=1:m
    Y(i, y(i)) = 1;
end

output = (-1/m).*(Y.*log(a_3) + (1-Y).*log(1-a_3));

trunTheta1 = Theta1(:,2:size(Theta1,2))(:);
trunTheta2 = Theta2(:,2:size(Theta2,2))(:);

J = sum(sum(output)) + (lambda/(2*m)) * (sum(trunTheta1'*trunTheta1) + sum(trunTheta2'*trunTheta2));


%a_1 = m x 401
%z_2 = m x 25
%a_2 = m x 26
%z_3 = m x 10
%a_3 = m x 10
%y = m x 1
%Y = m x 10
%Theta1 = 25 x 401
%Theta2 = 10 x 26
%delta_3 = 10 x 1
%delta_2 = 26 x 1

Accum_1 = zeros(size(Theta1));
Accum_2 = zeros(size(Theta2));

for i=1:m
    delta_3 = (a_3(i,:) - Y(i,:))';
    size(delta_3);
    Accum_2 = Accum_2 + delta_3*a_2(i,:); %=size(Theta2)

    delta_2 = (Theta2')*delta_3.*sigmoidGradient([1 z_2(i, :)]');
    size(delta_2);
    Accum_1 = Accum_1 + delta_2(2:end)*a_1(i,:); %=size(Theta1)
end


regular1 = [zeros(size(Theta1, 1),1) Theta1(:,2:size(Theta1, 2))];

Theta1_grad = Theta1_grad + (1/m).*Accum_1 + (lambda/m).*regular1;

regular2 = [zeros(size(Theta2, 1),1) Theta2(:,2:size(Theta2, 2))];
Theta2_grad = Theta2_grad + (1/m).*Accum_2 + (lambda/m).*regular2;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
