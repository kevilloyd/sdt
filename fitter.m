function [fit,ret] = fitter(belief,ps,n)
% INPUTS
% belief: the real belief state(s)
% ps: all of our represented belief states (i.e., discrete approx. to true continuous belief state)
% n: number of discrete belief states to interpolate between
% OUTPUTS
% fit: column 1 has the indices of the n closes belief points; column 2 is the vector of weights/distances
% ret: distance between weighted beliefs and true belief (i.e., the error in the approximation)

[~,b]=mink(sum(abs(ps-belief),2),n);

[x, w] = fnnls( ps(b,:)*ps(b,:)', ps(b,:)*belief' );
fit=[b x./sum(x)];
ret = sum(w);
