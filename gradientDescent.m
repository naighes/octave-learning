% EXAMPLE:
%
%     |1 1|
% X = |1 2|
%     |1 3|
%
%     |1|
% y = |2|
%     |3|
%
% iterations = 8000
%
% alpha = 0.01
function theta = gradientDescent(X, y, iterations, alpha)

    % X is the design matrix containing our training examples
    % y is the class labels

    % size(X) returns (rows, columns) pair
    % size(X, 1) returns the first dimension of the matrix (rows)
    m = size(X, 1);

    % initialize theta vector
    theta = zeros(size(X, 2), 1);

    for iter = 1:iterations
        errorVector = (X * theta) - y;
        theta = theta - ((1/m) * errorVector' * X)' * alpha;
    end
end

