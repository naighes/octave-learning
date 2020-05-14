% Computing cost and gradient for logistic regression
function [J, grad] = costFunction(theta, X, y)

    % X is the design matrix containing our training examples
    % y is a vector which elements are 0 or 1 (it's a binary classification)

    % number of training examples
    m = length(y);

    % J is the cost
    J = 0;

    % grad is the gradient
    grad = zeros(size(theta));

    % it's a logistic regression: calculate the sigmoid
    h = sigmoid(X * theta);
    J = (1/m) * (-y' * log(h) - (1 - y)' * log(1 - h));
    grad = (1 / m) * X' * (h - y);
end

