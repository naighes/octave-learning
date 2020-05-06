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
% theta = |0|
%         |1|
%
function J = costFunctionJ(X, y, theta)

    % X is the design matrix containing our training examples
    % y is the class labels

    % size(X) returns (rows, columns) pair
    % size(X, 1) returns the first dimension of the matrix (rows)
    m = size(X, 1);

    predictions = X * theta; % get prediction of all hypothesis on all m examples
    sqrErrors = (predictions - y).^2; % find the squared error
    J = 1/(2*m) * sum(sqrErrors);
endfunction

