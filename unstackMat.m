function [ apprx_mat ] = unstackMat(stackedApprxMat)
% This code unstacks the stacked vector to matrix for each layer

global N_layer;     global mat_size;    global network_arch;
inputSize   = network_arch.inputSize;
hiddenSize  = network_arch.hiddenSize;
outputSize  = network_arch.outputSize;

range = cell(1,N_layer);
range_offset = 0;

range{1} = [1 : inputSize*hiddenSize(1)];
for i = 1:N_layer-1
    range_offset = range_offset + length(range{i});    
    if (i ~= N_layer-1)
        range{i+1} = [range_offset+1 : range_offset+hiddenSize(i)*hiddenSize(i+1)];
    else
        range{i+1} = [range_offset+1 : range_offset+hiddenSize(i)*outputSize];
    end
end

apprx_mat = cell(1,N_layer);
for i = 1:N_layer
	apprx_mat{i} = reshape(stackedApprxMat(range{i}), mat_size(i,1), mat_size(i,2));
end


end

