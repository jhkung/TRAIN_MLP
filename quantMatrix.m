function [ quantizedMat ] = quantMatrix(inMat, quant_mat, prec)
% this code quantizes (approximates) input matrix with given precision (prec) 


% quantize an input matrix
quantizedMat = zeros(size(inMat));
quantizedMat = not(quant_mat) .* inMat + quant_mat .* float2fix(inMat, prec);



end

