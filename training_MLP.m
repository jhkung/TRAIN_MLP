function [ optTheta, grad, iter, cost ] = training_MLP(trainData, trainLabels, theta)
%% This code performs vectorized version of MLP forwardPass

global BENCHMARK;
global network_arch;
global target_cost;
inputSize   = network_arch.inputSize;
hiddenSize  = network_arch.hiddenSize;
outputSize  = network_arch.outputSize;
    
max_iter = 30000;
cost_arr = zeros(1,max_iter);
lrn_rate = 5;


if strcmp(BENCHMARK, 'MNIST')
    lrn_rate    = 15;
elseif strcmp(BENCHMARK, 'CNAE-9')
     lrn_rate   = 3;      %0.0118
elseif strcmp(BENCHMARK, 'SPAM')
    lrn_rate    = 0.4;
end



tic
sprintf('PRETRAINING BEGIN')
for iter = 1:max_iter
	if (mod(iter,100) == 0)
		iter
		cost
    end
    
    [cost, grad, hypothesis] = forwardPass(theta, inputSize, hiddenSize, outputSize, trainData, trainLabels);
    
	theta           = theta - lrn_rate * grad;
	cost_arr(iter)  = cost;
    
    if(cost < target_cost)
        sprintf('END ITERATION')
        cost/outputSize
        iter
        break;
    end
    
end
sprintf('PRETRAINING END')
toc

optTheta = theta;

end
