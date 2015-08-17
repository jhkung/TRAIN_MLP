function [ mat_size ] = setMatrixSize(inputSize, hiddenSize, outputSize)
% This code outputs matrix size of each weight and bias

global N_layer;
mat_size = zeros(2*N_layer,2);

% matrix size of weight matrices in each layer
mat_size(1,:) = [hiddenSize(1), inputSize];
for i = 1:N_layer-2
    mat_size(i+1,:) = [hiddenSize(i+1), hiddenSize(i)];
end
mat_size(N_layer,:) = [outputSize, hiddenSize(N_layer-1)];

% matrix size of bias vectors in each layer
for bias_idx = (N_layer + 1):1:(2 * N_layer - 1)
    mat_size(bias_idx,:) = [hiddenSize(bias_idx-N_layer), 1];
end
mat_size(2*N_layer,:) = [outputSize, 1];


end
