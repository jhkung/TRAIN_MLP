function [ Pcurr ] = computePower( rPrec, rApprx, flag )
% This code computes the overall power consumption of MLP MAC units

global Pow;

if (flag == 0)
    Pcurr = rPrec*Pow(4) + (rApprx - rPrec)*Pow(3) + (1 - rApprx)*Pow(1);
else
    Pcurr = rApprx*Pow(4) + (rPrec - rApprx)*Pow(2) + (1 - rPrec)*Pow(1);
end



end

