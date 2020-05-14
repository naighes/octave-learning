function g = sigmoid(z)

    % z can be a matrix, a vector or a scalar value.

    g = zeros(size(z));
    g = 1 ./ (1 + e .^ (-1 * z));
end

