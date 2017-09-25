function drawPredicationLine(X, theta)

    % plot the predication line
    figure(9999, "position", [550 100 560 420]);
    plot(X(:,2), X*theta, 'o', "markersize", 1)


end

