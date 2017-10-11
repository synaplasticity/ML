    %% Initialization
    clear ; close all; clc


    % load training data
    fprintf('Loading training data ... \n');
    data = load('ex1data1.txt');
    
    % Determine training data set length
    m = length(data);
    fprintf('Training data length (m) : %d : \n', m);

    
    % X = input data. Create a matrix as [1, training_data_input]
    fprintf('Creating training data input vector (X)... \n');
    X = [ones(m, 1), data(:,1)];
    
    % y = training data expected output
    fprintf('Creating training data expected value vector (y)... \n');
    y = data(:,2);
    
    % theta - intial weights as 1 dimensial vector representing theta_0 and
    % theta_1
    theta = zeros(2, 1);
    fprintf('Using initial weights (theta) as %d, %d : \n', theta);

    % alpha = GD learning rate
    alpha = 0.01
    
    % How many iterations
    num_iters = 500;
    fprintf('Iterations (num_iters) : %d : \n', num_iters);


