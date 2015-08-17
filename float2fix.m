function [fix] = float2fix(input, prec_bitwidth)
% Conversion from floating point number to fixed point number

if (prec_bitwidth(1) == -1)
    % floating point mode
    fix = input;
else
    
    
    num_bit     = prec_bitwidth(1);
    num_frac    = prec_bitwidth(2);
    
    max_prec    = (pow2(num_bit-1)-1) / pow2(num_frac);     % maximum value a given [num_bit] can handle
    min_prec    = -pow2(num_bit-1) / pow2(num_frac);        % minimum value a given [num_bit] can handle
    
    input_bin   = floor(input .* pow2(num_frac));
    fix         = input_bin ./ pow2(num_frac);
    
    fix(fix > max_prec) = max_prec;
    fix(fix < min_prec) = min_prec;
    
    % err = abs(input-fix);
    % prec_err = max(max(err));
end

end

