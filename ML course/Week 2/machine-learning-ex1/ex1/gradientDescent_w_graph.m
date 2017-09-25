function [theta, J_history] = gradientDescent_w_graph(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

format long;

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

fprintf("\nFeature size is %f : ", m);

%-----------------

    % Grid over which we will calculate J
    theta0_vals = linspace(-10, 10, 100);
    theta1_vals = linspace(-1, 4, 100);

    % initialize J_vals to a matrix of 0's
    J_vals = zeros(length(theta0_vals), length(theta1_vals));

    
    % Fill out J_vals
    for i = 1:length(theta0_vals)

        for j = 1:length(theta1_vals)
          t = [theta0_vals(i); theta1_vals(j)];
          J_vals(i,j) = computeCost(X, y, t);

          % fprintf("Theta =%0.15f",   J_vals(i,j));
          
        end
    end


    % Because of the way meshgrids work in the surf command, we need to
    % transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals';
    % Surface plot
%     figure;
%     surf(theta0_vals, theta1_vals, J_vals)
%     xlabel('\theta_0'); ylabel('\theta_1');

%     figure(9999);
%     plot(X, y, 'kx', 'Markersize', 5);
%     ylabel('Profit in $10,000s');
%     xlabel('Population of City in 10,000s');

    % Contour plot
    figure(999);
    % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    xlabel('\theta_0'); ylabel('\theta_1');
    hold on;




%-----------------



    plot(theta(1,:), theta(2,:), 'rx', 'MarkerSize', 20, 'LineWidth', 2);

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
        %fprintf("\nIteration %d, Theta =%0.15f;%0.15f, \ncost =%0.15f \n",  iter, theta, cc); 


        % plot the predication line
        figure(9999);
        plot(X(:,2), X*theta, 'o')

        % alternate between red and yellow color for QD values
        figure(999);
        if (mod(iter, 25) == 0)
            plot(theta(1,:), theta(2,:), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        elseif (mod(iter, 50) == 0)
             plot(theta(1,:), theta(2,:), 'yx', 'MarkerSize', 10, 'LineWidth', 2);            
        end

        
    end

    % Plot the final theta value
    plot(theta(1,:), theta(2,:), 'gx', 'MarkerSize', 20, 'LineWidth', 2);

    % plot the training data distribution
    figure(9999);
    plot(X, y, 'kx', 'Markersize', 5);
    ylabel('Profit in $10,000s');
    xlabel('Population of City in 10,000s');
    hold on;

    % plot the prediction line for the given data using the new theta
    % values
    plot(X(:,2), X*theta, '-')

end
