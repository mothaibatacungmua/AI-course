function g = sigmoid( z )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

g = 1.0 ./(1.0 + exp(-z));
end

