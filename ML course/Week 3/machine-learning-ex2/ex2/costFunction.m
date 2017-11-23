function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% cost function
% J(θ)=1m⋅(−yTlog(h)−(1−y)Tlog(1−h))
%

% y' is 1x100 vector.
% "y' * log(sigmoid(exponent))'" - transpose the log bit so we have 100x1 vector. 
%   (1-y)1 is 1x100 vector. So, "* log(1 - sigmoid(exponent))'" transpose 
%   the log(1-sig..), so we get 100X1 vector

J = getLogisticCost(theta, X, y);



% first transpose X for sigmoid function, so we can multiply (1x3)*(3x100) = (1x100).
% Transpose y (100X1) to (1x100) before. We will have a (1x100) vector (step 2)
%
%   Now transpose X, so we get 3X100 vector
%   Transpose step2 output so we get 100x1 vector
%       Multiplication of these two parts should provide 3x1 vector -> size of
%       theta
grad = gradient(theta, X, y);





% =============================================================

end
