function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
    % =============================================================


    %
    % Cost function J(theta)
    %
    % As theta0 is not regularized, we skip it
    cf_reg_expr = lambda/(2*m) * sum(theta([2 : size(theta)]) .^ 2);


    J = getLogisticCost(theta, X, y) + cf_reg_expr;

    %
    % Gradients
    %

    % As theta0 is not regualrized, we do not apply lambda factor
    grad = gradient(theta, X, y);
    grad_one = grad(1); % Store theta0 value as it's not regularized.

    % Regularization expression for non theta0 values
    gd_reg_expr = lambda/m * theta;

    % Add regularization for all values of theta(weights), but we will replace
    % theta0 with the non regularized value we caluclated earlier.

    grad =  grad + gd_reg_expr;

    % As theta0 is not regularized, we replace it with the non regularized
    % value
    grad(1) = grad_one;

end
