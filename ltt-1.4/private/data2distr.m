function [xx, D, N] = data2distr(d);
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).
eps = -1.0;

N = size(d.x, 2);   % number of data points
D = size(d.x, 1);   % number of variables

size(d.nsyms)

% fill 'xx' for the observed variables
xx = cell(D, 1);

for i = 1:D
  xx{i} = zeros(d.nsyms(i), N);
  for n = 1:N
    xx{i}(d.x(i, n), n) = 1.0;
  end
  if eps > 0
    xx{i} = (1 - eps) * xx{i} + eps / cast(d.nsyms(i), 'single');
  end
end
