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



X = [ones(m,1) X];

D1 = zeros(size(Theta1,1),1);
D2 = zeros(size(Theta2,1),1); 


for i = 1:m

alpha_1 = X(i,:);
%alpha_1 = [1 alpha_1];

zeta_2 = alpha_1 * Theta1';
alpha_2 = sigmoid(zeta_2);

alpha_2 = [1 alpha_2];


zeta_3 = alpha_2 * Theta2';
alpha_3 = sigmoid(zeta_3);

Y = zeros(num_labels,1);
Y(y(i)) = 1;

A = log(alpha_3)*(-Y);
B = log(1-alpha_3)*(1-Y);

J = J + A-B;




delta_3 = (alpha_3'-Y);
N = sigmoidGradient(zeta_2);
Q = Theta2' * delta_3;
Q = Q(2:end);
delta_2 = Q .* N';

D2 = D2 + delta_3 * alpha_2;
D1 = D1 + delta_2 * alpha_1;

%tri_2 = tri_2 + delta_3 * alpha_2';
%tri_3 = 

endfor

J = J/m

T1 = Theta1;
T1(:,1) = 0;
T1u = T1(:);

T2 = Theta2;
T2(:,1) = 0;
T2u = T2(:);

R = lambda/(2*m) * (T1u'*T1u + T2u'*T2u);


J = J + R;

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

disp('===');



Theta1_grad = D1/m;
Theta1_ = [zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta1_grad = Theta1_grad + (lambda / m) * Theta1_;
%Theta1_grad(1) = D1(1)/m;

disp(size(zeros(size(Theta1,1), 1)));
disp(size(Theta1_));


disp('-');
Theta2_grad = D2/m;

Theta2_ = [zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * Theta2_;
%Theta2_grad(1) = D2(1)/m;
disp(size(Theta2));
disp(size(Theta2_));



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients



grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
