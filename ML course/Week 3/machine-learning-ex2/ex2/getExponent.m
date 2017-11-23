function [exponent] = getExponent(theta, X)
    %
    % Retruns the exponent thetTx, whic is used as input to the sigmoid
    % function used on logistice regression
    %

    % The formula is thetaTx. However, we need to transpose X, so we can
    % successfuly multiply.
    exponent = theta'*X';

end