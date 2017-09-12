function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

format long

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

fprintf("\nFeature size is %f : ", m);

    for iter = 1:num_iters

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %

        i = 1;
        x0_errDiffSum = 0;
        x1_errDiffSum = 0;
        while (i <= m)
            x_i = X(i, :); % x(i). Row of X. E.g.: [1, 1.6]
%             fprintf("\nTraining set %d, value %f,%f", i, x_i);
    
            %fprintf("\nx_i is %f and y_i is %f ", x_i, y(iter));
    
            hypoXi = x_i * theta; % % h_theta(x_i)
    
            x0_errDiff = (hypoXi - y(i)); % x_0 is 1
            x1_errDiff = (hypoXi - y(i)) * x_i(:, 2); % Thing to sum in GD formula
    
            x0_errDiffSum = x0_errDiffSum + x0_errDiff;
            x1_errDiffSum = x1_errDiffSum + x1_errDiff;
                   
            i++;
        end

        theta(1,:) = ( theta(1,:) - (alpha * (1/m) * x0_errDiffSum) )  ; % final GD formula
        theta(2,:) = ( theta(2,:) - (alpha * (1/m) * x1_errDiffSum) )  ; % final GD formula


%         hypothesis_x = X * theta;
%         errDiff = (hypothesis_x - y) .* X(:, 2);
%         errDiffSum = sum(errDiff);
% 
%         theta = theta - (alpha * (errDiffSum / m));

        % ============================================================

        % Save the cost J in every iteration    
        %J_history(iter) = computeCost(X, y, theta);
        cc = computeCost(X, y, theta);
        J_history(iter) = cc;

        %fprintf("\nTheta %f Cost %f\n",  theta, J_history(iter)); 
        fprintf("\nIteration %d, Theta =%0.15f;%0.15f, \ncost =%0.15f \n",  iter, theta, cc); 

    end

end
