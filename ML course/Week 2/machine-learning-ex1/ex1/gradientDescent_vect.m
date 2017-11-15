function [theta, J_history] = gradientDescent_vect(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

clock;

format long

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

fprintf("\nFeature size is %f : ", m);

    for iter = 1:num_iters

        % We are going to treat the GD formula as theta = theta - alpha &
        % delta, where theta is n+1 vector, alphas is a scalar and delta is
        % n+1 vector
        % delta = 1/m SUM(h(theta) - y)*x for m training values

        % Transpose theta as per formula (h(theta) = theta' * x)
        % Note: The reason why we do this is to get an accumulate sum using
        % linear algebra
        % X' = transpose X, so that we have x0 and x1 as columns. [1; 6.7][1, 3.4] 
        %% step1 = theta' * X'; 
        % Part of GD formula to subtract y values
        %% step2 = step1 - y';
        % Part of GD formula to multiply by features (X). Note: We use
        % original X and due to step2 size(1, 97) and X sixe (97, 1), we have
        % summation
        %% step3 = step2 * X;
        % Patr of GD formula to divide by training size. Remember to
        % transpose step3 from size (1, 2) to (2, 1), which is the theta
        %  vector size
        %% step4 = step3' / m;
        
        %% theta = theta - (alpha * step4);
        
        delta = 1/m * (((theta' * X') - y') * X)';
        theta = theta - (alpha * delta);

        
        % Save the cost J in every iteration    
        %J_history(iter) = computeCost(X, y, theta);
        cc = computeCost(X, y, theta);
        J_history(iter) = cc;

        %fprintf("\nTheta %f Cost %f\n",  theta, J_history(iter)); 
        %fprintf("\nIteration %d, Theta =%0.15f;%0.15f, \ncost =%0.15f \n",  iter, theta, cc); 

    end
    
    clock;
end
