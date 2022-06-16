function tau_mat = get_belief_momdp( imats, a_opt, O, start, stopper, p_decay )

% derive the belief transition function. I.e., for a given initial belief
% at time t, and action a_t, what is the distribution over beliefs at t+1;
% but now when the signal is on between start and stopper
% and where a_opt is the optimal attentional action
%
% INPUTS
% imats: the interpolation function given a belief state and a particular observation
% a_opt: the optimal policy
% O: observation function
% start = when the signal comes on (or N+1 if it doesn't)
% stopper = when the signal goes off (or N+1 if it doesn't)
% OUTPUTS
% tau_mat: the transition matrix for belief states (b x b')

[nps, nx, N, interp, ~] = size(imats{1}); 

onoff=zeros(1,N);
if (start<=N)
    onoff(start:min(N,stopper))=1;
end

% this is the same as in the optimization loop
% except that we use the signal presence function
% to tell us the probability of a signal
tau_mat = zeros(2*nps,2*nps,N);
idx = zeros(nx,interp);
weights = zeros(nx,interp);
for n=1:N-1 % for all time steps except the final decision state
    for i=1:2*nps
        act=a_opt(i,n);
        ii=1+mod(i-1,nps);
        idx(:,:) = imats{act}(ii,:,n,:,1);
        weights(:,:) = imats{act}(ii,:,n,:,2);
        tau_mat(i,(act-1)*nps+(1:max(idx(:))),n) = tau_mat(i,(act-1)*nps+(1:max(idx(:))),n) + accumarray( idx(:), weights(:).*repmat(O{1+onoff(n),act}',interp,1) )'; 
    end
end
% what about the decision state (where there are no observations)?
tau_mat(a_opt(:,N)==1,nps,N) = 1; % if choose WEAK, then we know exactly where one will end up at the start of the next trial
idx_s = find(a_opt(:,N)==2); % the belief states for which choose STRONG
tau_mat(idx_s,nps,N) = p_decay; % if choose STRONG, there is small probability p_decay of starting next trial in WEAK state
tau_mat(idx_s,nps*2,N) = 1-p_decay; 
