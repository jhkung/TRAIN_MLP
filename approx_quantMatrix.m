function [ quantizedMat ] = approx_quantMatrix(inMat, quant_mat, approx_mat, prec)
% This code quantizes (approximates) theta (weights) of MLP

%global prec;
global recovery;
global err_data;
% quantization matrix
quantizedMat = zeros(size(inMat));
quantizedMat = not(quant_mat) .* float2fix(inMat, [32, 24]) + quant_mat .* float2fix(inMat, prec);

% Add approximation error
[r c] = size(inMat);
num_err= r*c;

% 1. Bring error vector for 32-bit (no precision controlled)
[err_vec_non_prec] = err_vec_gen(num_err, recovery, 32);
% 2. Bring error vector for given precision controlled 
[err_vec_with_prec] = err_vec_gen(num_err, recovery, prec(1));

err_mat_non_prec = reshape(err_vec_non_prec, size(inMat)).*approx_mat;
err_mat_with_prec = reshape(err_vec_with_prec, size(inMat)).*approx_mat;

quantizedMat = quantizedMat + not(quant_mat) .* err_mat_non_prec + quant_mat .* err_mat_with_prec;
%prec    = [16 8; 16 8];         % precision for operand and weight


end

