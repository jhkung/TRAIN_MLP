clc;
clear all;
global train_prec;
train_prec = [32 24];


%% For Duckhwan Kim (Local Machine)
% addpath C:\Users\jhkung\Documents\GitHub\RAW_DATA
% addpath C:\Users\jhkung\Desktop\RAW_DATA
addpath ../RAW_DATA
%%

num_iter_for_avg = 10;
while train_prec(1) >= 16
    global result;
    result.benchmark = '';
    result.train_prec = [0 0];
    result.cost_vec = [];
    result.iter_vec = [];
    result.accuracy_vec = [];
    result.grad = [];
    result.opt_theta =[];
    result.avg_iter = 0;
    result.avg_cost = 0;
    result.avg_accuray = 0;
    
    for i=1:num_iter_for_avg
        if (mod(i,10)==0) 
            i
            train_prec
        end
        MLP_sim;
    end
    result.avg_iter = mean(result.iter_vec);
    result.avg_cost = mean(result.cost_vec);
    result.avg_accuray = mean(result.accuracy_vec);
    fname = sprintf('%s_%d_%d.mat',result.benchmark,train_prec(1),train_prec(2));
    sprintf('SAVING RESULT ... %s',fname)
    save(fname,'result');
    train_prec = train_prec - [4 4];
end