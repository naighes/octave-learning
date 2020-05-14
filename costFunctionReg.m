% Computing cost and gradient for logistic regression with regularization
function [J, grad] = costFunctionReg(theta, X, y, lambda)

    % X is the design matrix containing our training examples
    % y is a vector which elements are 0 or 1 (it's a binary classification)
    m = length(y); % number of training examples
    J = 0;
    grad = zeros(size(theta));

    % the first element of theta is replaced by 0
    theta_r = [0; theta(2:size(theta))];

    [J, grad] = costFunction(theta, X, y);
    J = J + (lambda / (2 * m)) * (theta_r' * theta_r);
    grad = grad + (lambda / m) * theta_r;
end

