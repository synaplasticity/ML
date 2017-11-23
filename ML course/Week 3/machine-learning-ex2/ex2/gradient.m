function[grad] = gradient(theta, X, y)

    %
    % Returns the gradient for logistic regression
    % 

    m = length(y); % training sample size
    
    exponent = getExponent(theta, X);
    
    % first transpose X for sigmoid function, so we can multiply (1x3)*(3x100) = (1x100).
    % Transpose y (100X1) to (1x100) before. We will have a (1x100) vector (step 2)
    %
    %   Now transpose X, so we get 3X100 vector
    %   Transpose step2 output so we get 100x1 vector
    %       Multiplication of these two parts should provide 3x1 vector -> size of
    %       theta

    grad = 1/m * ( X' * (sigmoid(exponent) - y')' );

end