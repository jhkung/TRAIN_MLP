function [ cost, hypothesis ] = prec_apprx_forwardPass(theta, inputSize, hiddenSize, outputSize, data, label, prec_mat, apprx_mat, mode)
tic
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, ..., b1, b2, ...) matrix/vector format

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
% range{2}.w = [length(range{1}.w)+1 : length(range{1}.w)+hiddenSize(1)*hiddenSize(2)];
% range{3}.w = [length(range{1}.w)+length(range{2}.w)+1 : length(range{1}.w)+length(range{2}.w)+hiddenSize(2)*outputSize];


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
% range{2}.b = [tmp_range+length(range{1}.b)+1 : tmp_range+length(range{1}.b)+hiddenSize(2)];
% range{3}.b = [tmp_range+length(range{1}.b)+length(range{2}.b)+1 : tmp_range+length(range{1}.b)+length(range{2}.b)+outputSize]; 

% convert vector into matrix format
cost = 0;
synapse = cell(1,N_layer);
for i = 1:N_layer
	synapse{i}.w = reshape(theta(range{i}.w), mat_size(i,1), mat_size(i,2));
	synapse{i}.b = theta(range{i}.b);
end

%% Forward Pass
num_training   = size(data, 2);
x   = data;     y   = label;
a{1}  = x;      % input to the 1st layer

global prec;

%% input layer to 1st hidden layer, 1st to 2nd hidden layer, ..., (N_layer-2) to (N_layer-1) hidden layer
for layer_idx = 1:(N_layer-1)
    sprintf('COMPUTING HIDDEN LAYER (%d)...', layer_idx)
    z{layer_idx}    = [];
    qW{layer_idx}   = zeros(size(synapse{layer_idx}.w));
    
    qW{layer_idx} = qW{layer_idx} + approx_quantMatrix(synapse{layer_idx}.w, prec_mat{layer_idx}, apprx_mat{layer_idx}, prec);             % quantize the weight{layer_idx}
    
    for idx = 1:hiddenSize(layer_idx)
        qA{layer_idx} = zeros(size(a{layer_idx}));
        
        qA{layer_idx} = qA{layer_idx} + approx_quantMatrix(a{layer_idx}, repmat(prec_mat{layer_idx}(idx,:)',1,num_training), repmat(apprx_mat{layer_idx}(idx,:)',1,num_training) ,prec);
        
        z{layer_idx}	= [z{layer_idx}; qW{layer_idx}(idx,:) * qA{layer_idx}];
    end
    
    z{layer_idx} = z{layer_idx} + repmat(synapse{layer_idx}.b,1,num_training);
    z{layer_idx} = quantMatrix(z{layer_idx}, ones(size(z{layer_idx})), [32, 24]);
    a{layer_idx+1} = sigmoid(z{layer_idx}, mode);
end


%% (N_layer-1) hidden layer to output layer
sprintf('COMPUTING OUTPUT LAYER...')
z{N_layer} 	= [];
qW{N_layer} = zeros(size(synapse{N_layer}.w));

qW{N_layer} = qW{N_layer} + approx_quantMatrix(synapse{N_layer}.w, prec_mat{N_layer}, apprx_mat{N_layer}, prec);             % Quantize the weight{1}

for idx = 1:outputSize
    qA{N_layer} = zeros(size(a{N_layer}));
          
    curr_prec_mat = prec_mat{N_layer}(idx,:)';
    
    qA{N_layer} = qA{N_layer} + approx_quantMatrix(a{N_layer}, repmat(curr_prec_mat,1,num_training),repmat(apprx_mat{N_layer}(idx,:)',1,num_training) ,prec);
    
    z{N_layer}	= [z{N_layer}; qW{N_layer}(idx,:) * qA{N_layer}];
end

z{N_layer} = z{N_layer} + repmat(synapse{N_layer}.b,1,num_training);
z{N_layer} = quantMatrix(z{N_layer}, ones(size(z{N_layer})), [32, 24]);
hypothesis = sigmoid(z{N_layer}, mode);

cost = mean(0.5 * sum((y - hypothesis) .^ 2));

toc
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x, mode)
  
    if (mode == 0)
        sigm = 1 ./ (1 + exp(-x));
    else
        sigm = 1/24 * (abs(x+4) + abs(x+2) - abs(x-2) - abs(x-4)) + 0.5;
    end
    
end

function deriv_sigm = deriv_sigmoid(x, mode)

    if (mode == 0)
        deriv_sigm = sigmoid(x, mode) .* (1-sigmoid(x, mode));
    else
        deriv_sigm = sigmoid(x, mode) .* (1 - sigmoid(x, mode));
    end
    
end
