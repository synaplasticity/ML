function drawFinalPredictionGraph(X, y, theta)

    % plot the training data distribution 
    %figure(9999, "position", [500 300 560 420]);
%     figure(9999);
%     plot(X, y, 'kx', 'Markersize', 5);
%     ylabel('Profit in $10,000s');
%     xlabel('Population of City in 10,000s');
%     hold on;
    
    % plot the prediction line on the given data using the new theta
    % values
    figure(9999);
    plot(X(:,2), X*theta, '-', "markersize", 20)
    
end
