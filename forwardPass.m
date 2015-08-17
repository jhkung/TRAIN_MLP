function [cost,grad,hypothesis] = forwardPass(theta, inputSize, hiddenSize, outputSize, data, label)
                                   
                                              
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format

global mat_size;	global N_layer;
range           = cell(1,N_layer);
range_offset    = 0;

range{1}.w = [1 : inputSize*hiddenSize(1)];
for i = 1:N_layer-1
    range_offset = range_offset + length(range{i}.w);    
    if (i ~= N_layer-1)
        range{i+1}.w = [range_offset+1 : range_offset+hiddenSize(i)*hiddenSize(i+1)];
    else
        range{i+1}.w = [range_offset+1 : range_offset+hiddenSize(i)*outputSize];
    end
end


tmp_range   = length(range{1}.w);
for L_idx = 1:N_layer-1
    tmp_range = tmp_range + length(range{L_idx+1}.w);
end
range{1}.b = [tmp_range+1 : tmp_range+hiddenSize(1)];

range_offset = 0;
for i = 1:N_layer-1
    range_offset = range_offset + length(range{i}.b);    
    if (i ~= N_layer-1)
        range{i+1}.b = [tmp_range+range_offset+1 : tmp_range+range_offset+hiddenSize(i+1)];
    else
        range{i+1}.b = [tmp_range+range_offset+1 : tmp_range+range_offset+outputSize];
    end
end

% Added by Duckhwan Kim (150716)
%1. Training bit precision [num_bit, num_bit_for_fractional_part]
% float2fix(input, prec_bitwidth)
global train_prec;

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros.
% stacked vector to matrix conversion
cost = 0;
synapse = cell(1,N_layer);
for i = 1:N_layer
	synapse{i}.w = float2fix(reshape(theta(range{i}.w), mat_size(i,1), mat_size(i,2)),train_prec);
	synapse{i}.b = float2fix(theta(range{i}.b),train_prec);

	synapse{i}.wgrad = zeros(size(synapse{i}.w));
	synapse{i}.bgrad = zeros(size(synapse{i}.b));
end


%% Forward Pass
num_training   = size(data, 2);
x   = data;     y   = label;
a{1} 	= x;

%% input layer to 1st hidden layer, 1st to 2nd hidden layer, ..., (N_layer-2) to (N_layer-1) hidden layer
for layer_idx = 1:N_layer
%     sprintf('COMPUTING HIDDEN LAYER (%d)...', layer_idx)
    z{layer_idx} = [];
    
    z{layer_idx} = float2fix(synapse{layer_idx}.w * a{layer_idx} + repmat(synapse{layer_idx}.b, 1, num_training),train_prec);
    a{layer_idx+1} = float2fix(sigmoid(z{layer_idx}),train_prec);    
end

hypothesis = a{N_layer+1};


%% Backpropagation: gradient computation
delta       = cell(1,N_layer);
J_W_grad    = cell(1,N_layer);
J_b_grad    = cell(1,N_layer);

delta{N_layer}      = float2fix((a{N_layer+1} - y) .* deriv_sigmoid(z{N_layer}),train_prec);
J_W_grad{N_layer}   = float2fix(delta{N_layer} * a{N_layer}',train_prec);
J_b_grad{N_layer}   = float2fix(delta{N_layer},train_prec);

for L_idx = (N_layer-1):-1:1
    delta{L_idx}    = float2fix((synapse{L_idx+1}.w' * delta{L_idx+1}) .* deriv_sigmoid(z{L_idx}),train_prec);
    J_W_grad{L_idx} = float2fix(delta{L_idx} * a{L_idx}',train_prec);
    J_b_grad{L_idx} = float2fix(delta{L_idx},train_prec);
end


% average over batch training set
for idx = 1:N_layer
    synapse{idx}.wgrad  = float2fix((J_W_grad{idx}/num_training),train_prec);
    synapse{idx}.bgrad  = float2fix(mean(J_b_grad{idx}')',train_prec);
end


%% Cost function (sum of squared error)
cost = mean(0.5 * sum((y - hypothesis) .^ 2)); 


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
grad = [];

for idx = 1:N_layer
    grad    = [grad; synapse{idx}.wgrad(:)];
end

for idx = 1:N_layer
    grad    = [grad; synapse{idx}.bgrad(:)];
end

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
    
end

function deriv_sigm = deriv_sigmoid(x)

    deriv_sigm = sigmoid(x) .* (1-sigmoid(x));
    
end
