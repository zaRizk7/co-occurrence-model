function atree = bin_forrest(data, opt);
% estimates a latent forrest of arbitrary trees using various criteria
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

repeats = 1000;
pvalue = 0.99;
if ~exist('opt', 'var')
  opt = []; 
end
if ~isfield(opt, 'verbose')
  opt.verbose = 0;
end
if all(data.nsyms==2), 
  opt.Kmax = 2; 
  fprintf('[%s.m] opt.Kmax=2;\n', mfilename);
end
if ~isfield(opt, 'signed')
  opt.signed = 0;
end
if opt.verbose > 1
  fprintf('[%s.m] chosen options (fields of struct "opt"):\n', mfilename);
  disp(opt);
end

[xx, D, N] = data2distr(data);
opt.N = N;
opt.D = D;
betas = xx;  % the initial beta messages are just the xx
% D == number of leaves
% N == number of data points

nsyms = data.nsyms;        % number of symbols for each variable

t0 = [];                   % the set of roots
p0 = {};                   % the distributions of the roots
t = cell(D, 1);            % the kids of each node (also of the leaves)
p = cell(D, 1);            % the conditional probability distributions
localbic = zeros(D, 1);        % store the local overall BIC
localbic_diff = zeros(D, 1);   % store the local overall BIC improvement

if opt.verbose > 1
  fprintf('[%s.m] calculating mutual information...\n', mfilename)
end
m = mi_mat(data.x, nsyms, [], [], opt.verbose, opt.signed);    % pairwise mutual information
if opt.verbose > 1
  fprintf('[%s.m] done\n', mfilename);
end
m(1:(D+1):end) = -Inf;     % ignore diagonal
if opt.verbose > 1
  m                          % print out the MI matrix
end

msg = sprintf('[%s.m] bug detected', mfilename);
frontier = ones(1, D);     % which nodes are at the frontier
localbic_prev = bic(data, thetree(frontier, xx, N, t, p, D, nsyms));
next = D+1;                % the index of the next latent variable
while 1
  % (1) find next candidate pairs
  nvars = size(m, 2);
  if next ~= nvars + 1
    error(msg);
  end
  [dummy, idx] = max(m(:)); % find maximal value
  if dummy == -Inf
    % no more nodes to be merged, we should stop elsewhere
    error(msg);
  end
  i = mod(idx-1, nvars)+1;  % row index of maximum
  j = ceil(idx/nvars);      % col index of maximum
  if i==j
    error(msg);
  end
  % note that i and j are only candidates
  
  % (2) learn new LCM for i and j
  kids = [i j];
  if opt.verbose > 0
    fprintf('[%s.m] candidate nodes %d and %d with MI==%f\n', mfilename, i, j, m(i,j));
  end
  K = 1;
  % run EM
  [CPTs, ll, K, opt] = em_automatic(betas, nsyms, kids, opt);
  beta_z = CPTs.qz_xi;  % the beta messsage for the new node
  % qz_xi can be calculated from beta_z
  pie = CPTs.pz;
  qz_xi = beta_z .* (pie * ones(1, size(beta_z, 2)));
  qz_xi = qz_xi ./ (ones(K, 1) * pie' * beta_z);
  betas{next} = beta_z;
  px_z  = CPTs.px_z;
  if K == 1
    % we are done, since the number of hidden states is one
    if opt.verbose == 1
      thet = thetree(frontier, xx, N, t, p, D, nsyms);
      thet.name = [data.name '_bin'];
      figure(1)
      subplot(122)
      tree2fig(thet)       % pure matlab
      %%% tree2dot(thet)   % might work but requires GraphViz
      title('estimate')
      disp(' ')
      fprintf('The nodes %d and %d have not been merged with a new latent node,\n', ...
              kids(1), kids(2))
      fprintf('since it would have only a single hidden state.\n');
      disp(' ')
      disp('Thus the algorithm is done and the right panel of Figure 1 shows')
      disp('the final tree structure.')
    end
    break
  end
  % (2) add the new latent variable and remove the candidates from the frontier

  % remove i and j from and add k to the frontier
  frontier(kids) = 0;
  frontier(next) = 1;
  
  xx{next} = qz_xi;
  nsyms(next) = K;
  if K~=size(qz_xi, 1)
    error(msg);
  end

  % add CPTs for the new latent variable
  p{next} = px_z;
  
  % add a link from 'next' to the kids
  t{next} = kids;   % take a note of the indices of the kids
  if opt.verbose > 0
    fprintf('[%s.m] add link %d -> %d and %d -> %d\n', mfilename, next, kids(1), next, kids(2));
  end
  if opt.verbose == 1  % demo mode
    thet = thetree(frontier, xx, N, t, p, D, nsyms);
    thet.name = [data.name '_bin'];
    figure(1)
    subplot(122)
    tree2fig(thet)       % pure matlab
    %%% tree2dot(thet)   % might work but requires GraphViz
    title('estimate')
    disp(' ')
    fprintf('The nodes %d and %d have been merged with the new latent node %d,\n', ...
            kids(1), kids(2), next)
    fprintf('that has K=%d hidden states.  K has been found by a binary search.\n', K)
    disp(' ');
    disp('The right panel of Figure 1 shows the current tree structure,')
    disp('which might grow further.')
    disp(' ');
  end
  if isfield(opt, 'orientmerge') && opt.orientmerge == 1
    % do orientation after each merge
    if opt.verbose > 1
      fprintf('[%s.m] orient after merge\n', mfilename);
    end
    thet = thetree(frontier, xx, N, t, p, D, nsyms);
    thet = orient(thet);
    t = thet.t;
    p = thet.p;
    t0 = thet.t0;
    p0 = thet.p0;
  end
  
  localbic(next) = bic(data, thetree(frontier, xx, N, t, p, D, nsyms));
  localbic_diff(next) = localbic(next) - localbic_prev;
  localbic_prev = localbic(next);
  
  nsyms(next) = K;
  
  % set the entries for i and j to -Inf to ensure they are ignored next time
  m(kids, :) = -Inf;
  m(:, kids) = -Inf;
  
  % calculate the mutual information between z and everybody else who is
  % still in the game
  m(next, :) = -Inf;
  m(:, next) = -Inf;   % set default values
  for ii = 1:(next-1)  
    if frontier(ii) > 0 & sum(kids==ii)==0
      % ii is still in the game
      % the contingency matrix for ii and k
      counts = xx{next} * xx{ii}';
      m(next, ii) = mi(counts/sum(counts(:)), opt.signed);
      m(ii, next) = m(next, ii);
    end
  end
  
  % increment the index of the next new variable
  if sum(frontier) < 2
    % we are done
    if opt.verbose == 1
      disp('However, since there are no more nodes to merge, we are done.')
      disp(' ')
      disp('Thus the algorithm has finished and the right panel of Figure 1 shows')
      disp('the final tree structure.')
    end
    break
  end
  if opt.verbose == 1
    disp('Press any key to continue.');
    pause
    clc
  end
  next = next + 1;
end

% create the tree structure
atree = thetree(frontier, xx, N, t, p, D, nsyms);
atree.localbic = localbic;
atree.localbic_diff = localbic_diff;
atree.xx = xx;
atree.name = [data.name '_bin'];
if isfield('opt', 'namesuffix')
  atree.name = [atree.name, opt.namesuffix];
end
atree.opt = opt;
check_forrest(atree);

% refinement
if 1
  atree.ll_before_refine = forrest_ll_fast(data, atree);
  old_ll = -inf;
  new_ll = atree.ll_before_refine;
  counter = 1;
  while new_ll - old_ll >= 1e-2
    atree = forrest_refine(data, atree);
    old_ll = new_ll;
    new_ll = forrest_ll_fast(data, atree);
    counter = counter + 1;
    if counter > 100, break, end
  end
  atree.ll_after_refine = forrest_ll_fast(data, atree);
end

if all(atree.nsyms==2)
  atree = orient(atree);   % sparsify the latent nodes
end
return


function atree = thetree(frontier, xx, N, t, p, D, nsyms);
% all nodes remaining in the frontier become roots
t0 = find(frontier);
p0 = cell(1, length(t0));
for i = 1:length(t0)
  root = t0(i);
  p0{i} = sum(xx{root}, 2)/N;
end
atree.t0 = t0;
atree.p0 = p0;
atree.t = t;
atree.p = p;
atree.nobs = D;
atree.nsyms = nsyms;
atree.df = degreesoffreedom(atree);
return
