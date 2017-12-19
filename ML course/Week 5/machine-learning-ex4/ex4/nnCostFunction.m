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
p = zeros(size(X, 1), 1);


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


% Add bias term to X
X = [ones(m, 1) X];
Theta1 = [ones(1, size(Theta1, 2)); Theta1]; % Add bias as the first row (1x401)


% convert numerals to logisitic classifier. E.g.: 5 = [0 0 0 0 1 0 0 0 0 0]
logisitic_y = [1:num_labels] == y;


for i = 1 : m
    
    %
    %% Feedforward propagation
    %
    
    % hidden layer (26x1)
    a_1 = sigmoid(Theta1 * X(i,:)');
    a_1(1) = 1; % (hidden layer zeroth unit) set a(2)(0) = 1 as bias unit.
    
    % output layer (10x1)
    output_layer_activation = sigmoid(Theta2 * a_1);
    [p(i,:), class] = max(output_layer_activation, [], 1);
    
    % calculate cost for ith activation unit(out_layer_activation) and ith output(y)
    % So in this 3 NN case, the output layer is a_3
    classification_cost_sum = (-logisitic_y(i,:) * log(output_layer_activation))...
                                - ((1 - logisitic_y(i, :)) * log(1 - output_layer_activation));
     
    J = J + classification_cost_sum; % Sum for each training data

    % Calculate delta3 (output layer) (10x26) same as Theta2
    % transpose logisitic_y (from col to row), so we subtract
    % 10 length boolean logisitic representation of numbers from 
    % 10 (k number of classes) output units
    delta3_i = (output_layer_activation - logisitic_y(i, :)');
    % + ((10x1) * (26x1))
    Theta2_grad = Theta2_grad + (delta3_i * (a_1'));

    % Calculate delta2 (Hidden layer). There is no delta1. (25x401) same
    % as Theta1
    %          (26x10 * 10x1)
    delta2_i = (Theta2' * delta3_i) .* sigmoidGradient(Theta1 * X(i,:)'); % z = Theta*x and a = signmoid(z). a is the activation function.
    delta2_i = delta2_i(2:end); % remove 0th delta element (25x1)
    Theta1_grad = Theta1_grad + (delta2_i * X(i, :)); % (25x1).(1x401)

end

J = J * (1/m); % complete the cost formula by dividing by training data size

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

%printf('\n\n**************T1 and T2 grad size : %d and %d', size(Theta1_grad), size(Theta2_grad));
% Add regularization term
% NOTE: The bias unit is not included for regularization
%   Theta1 is 26 X 401. We need to consider 25 X 401
%   Theta2 is 10 X 26. We need to consider 10 x 25
regularization_value = ( sum(sum(Theta1([2:end], [2:end]) .^ 2)) + ...
                             sum(sum(Theta2(:, [2:end]) .^ 2)) ) * ...
                                    (lambda / (2*m));


% Add it to the cost
J = J + regularization_value;

Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:, 2:end) + regularization_value];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:, 2:end) + regularization_value];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
