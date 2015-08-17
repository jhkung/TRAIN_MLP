function [ output ] = onehotEncode(labels)
%% This code provides desired output with one-hot encoding

global BENCHMARK;

if (strcmp(BENCHMARK, 'MNIST') || strcmp(BENCHMARK, 'CIFAR10'))
    output = zeros(10, length(labels));
elseif strcmp(BENCHMARK, 'LETTER')
    output = zeros(26, length(labels));
elseif strcmp(BENCHMARK, 'SPAM')
    output = zeros(2, length(labels));
end

for i = 1:length(labels)
    output(labels(i)+1, i) = 1;
end

end
