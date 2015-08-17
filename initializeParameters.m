function [ stackedTheta ] = initializeParameters(inputSize)
%% Initialize parameters randomly based on layer sizes

global mat_size;	global N_layer;
init_factor = 15;

% we'll choose weights uniformly from the interval [-r, r]
r               = zeros(1,N_layer);
synapse         = cell(1,N_layer);
stackedTheta    = [];

for i = 1:N_layer 
	r(i)            = sqrt(init_factor) / sqrt(mat_size(i,1)+inputSize+1);
	synapse{i}.w    = rand(mat_size(i,1), mat_size(i,2)) * 2 * r(i) - r(i);
	synapse{i}.b    = zeros(mat_size(i+N_layer,1), mat_size(i+N_layer,2));

	% convert weight and bias gradients to the vector form.
	stackedTheta = [stackedTheta ; synapse{i}.w(:)];
end

% convert weights and bias gradients to the vector form.
% this step will "unroll" (flatten and concatenate together) all your parameters into a vector
for i = 1:N_layer
    stackedTheta = [stackedTheta ; synapse{i}.b(:)];
end


end

