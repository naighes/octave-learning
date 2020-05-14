% Predicting whether the label is 0 or 1 using learned logistic
% regression parameters theta
function p = predict(theta, X)

    % X is the design matrix containing our training examples

    % number of training examples
    m = size(X, 1);
    p = zeros(m, 1);

    B = sigmoid(X * theta);
    p = B >= 0.5;
end

