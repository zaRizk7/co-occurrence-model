% ltt_demo3.m
% demonstrates ltt on dataset 3 "three-coins"
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).
close all
clear
opt.verbose = 1;  % for demonstrations
dataset = 3;
msg = sprintf('%d "THREE-COINS", which is generated by the\n', dataset);
msg = sprintf('%sfollowing tree structure:\n', msg);
msg = sprintf('%s\n', msg');
msg = sprintf('%s%s\n', msg, '          4      ');
msg = sprintf('%s%s\n', msg, '        / | \    ');
msg = sprintf('%s%s\n', msg, '       1  2  3   ');
msg = sprintf('%s\n', msg);
msg = sprintf('%sThe leaf variables x1, x2, x3 have each four states.  The root has eight states.\n', msg);
msg = sprintf('%sThe root encodes the combined values of three coins (three bits == 8 states).\n', msg);
msg = sprintf('%sEach of these eight states are equally likely.  Thus it corresponds to\n', msg);
msg = sprintf('%sflipping three coins and coding the result in a single integer ranging\n', msg);
msg = sprintf('%sfrom one to eight.  Each of the leaves is representing the combined state\n', msg);
msg = sprintf('%sof two of the coins, in a way such that each leaf is missing the\n', msg);
msg = sprintf('%sinformation from one of the three coins.  Thus the mutual information\n', msg);
msg = sprintf('%sbetween each pair of leaves is one bit.  For more details\n', msg);
msg = sprintf('%splease take a look at "create_tree.m".\n', msg);

msg_final = sprintf('Of course the THREE-COINS dataset is an example in which the \n');
msg_final = sprintf('%sgreedy binary learning fails, as can be seen in this demo.  The\n', msg_final);
msg_final = sprintf('%sreason is that the correct data structure with three children is\n', msg_final);
msg_final = sprintf('%snot part of the model.\n', msg_final);

% (0) introduction
clc
fprintf('[%s.m] starting demonstration\n', mfilename);
ltt_demo_helper