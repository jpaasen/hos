function [ula D] = getSimpleULA(no_elements, lambda, spacing, height, no_el_sub_elems)
% GETSIMPLEULA Returns an Uniform Linear Array centered around origin.
%
%   no_elements:            number of azimuth array elements
%   lambda:         
%   spacing:                spacing between elements in lambdas
%   height:                 element height in lambdas
%   no_el_sub_elems:        number of elevation points per element 
%
%   ula:                    coordinate for each point src. Dim: [no_elements, no_el, 3]
%   D:                      array width

D = ((no_elements-1) * lambda * spacing);

%H = lambda * height;
%el_points = [zeros(no_el_sub_elems,1), zeros(no_el_sub_elems,1), linspace(-H/2,H/2,no_el_sub_elems)'];

ula = [linspace(-D/2,D/2,no_elements); zeros(1,no_elements); zeros(1,no_elements);]';

end

