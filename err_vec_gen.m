function [err_vec] = err_vec_gen(num_err, recovery, prec)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%num_err = 10000;
%recover = 10;
%forcing_prec = 16;

if num_err ==0
    err_vec = 0;
else
    global err_data;
    data = err_data.Err_hist;
    prec_index = (32-prec)/4+1;
    recover_index = (recovery)/10;
    err_arr_orig=data(prec_index,recover_index).arr;
    
    
    err_vec = zeros(1,num_err);
    
    
    if num_err > length(err_arr_orig)
        rate = floor(num_err/length(err_arr_orig));
        for i=1:rate
            P = randperm(length(err_arr_orig));

            err_vec(1+ length(err_arr_orig)*(i-1):length(err_arr_orig)*i) = err_arr_orig(P);
        end
        P = randperm(length(err_arr_orig));

        err_vec(1+ length(err_arr_orig)*(rate): end) = err_arr_orig(P(1:num_err - rate*length(err_arr_orig)));
    else
        P = randperm(length(err_arr_orig));
        err_vec = err_arr_orig(P(1:num_err));
    end
    
    
end




end

