function [theta, J_history] = gradientDescent_w_graph(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

    format long;

    % Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1); % Compute cost per training data

    fprintf("\nFeature size is %f : ", m);


    % Create the initial contour plot and the spot with initial
    % theta values E.g. (0, 0)
    drawInitialContour(X, y);
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
    
            hypoXi = x_i * theta; % % h_theta(x_i)
    
            x0_errDiff = (hypoXi - y(i)); % x_0 is 1
            x1_errDiff = (hypoXi - y(i)) * x_i(:, 2); % Thing to sum in GD formula
    
            x0_errDiffSum = x0_errDiffSum + x0_errDiff;
            x1_errDiffSum = x1_errDiffSum + x1_errDiff;
                   
            i++;
        end

        % Simulatenously update theta values
        theta(1,:) = ( theta(1,:) - (alpha * (1/m) * x0_errDiffSum) )  ; % final GD formula
        theta(2,:) = ( theta(2,:) - (alpha * (1/m) * x1_errDiffSum) )  ; % final GD formula


        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);

        % plot the predication line
        figure(9999);
        plot(X(:,2), X*theta, 'o')

        % alternate between red and yellow color for QD values
        % so we have a better progressive change on the screen
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
    % plot the prediction line on the given data using the new theta
    % values
    plot(X(:,2), X*theta, '-')

end
