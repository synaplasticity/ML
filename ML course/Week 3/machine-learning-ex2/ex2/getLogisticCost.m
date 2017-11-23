function cost = getLogisticCost(theta, X, y)

    %
    % Returns the cost for logitic regression classifier by evaluating the
    % cost function
    % J(θ)=1m⋅(−yTlog(h)−(1−y)Tlog(1−h))
    % 
    
    m = length(y); % training sample size

    exponent = getExponent(theta, X);

    % y' is 1x100 vector.
    % "y' * log(sigmoid(exponent))'" - transpose the log bit so we have 100x1 vector. 
    %   (1-y)1 is 1x100 vector. So, "* log(1 - sigmoid(exponent))'" transpose 
    %   the log(1-sig..), so we get 100X1 vector

    cost = (1/m * (-(y' * log(sigmoid(exponent))') - ((1 - y)' * log(1 - sigmoid(exponent))')));

end