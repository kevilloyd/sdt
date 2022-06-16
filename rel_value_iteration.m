function [a_opt, r_av, v, Q_att, a_opt_ch] = rel_value_iteration( tau_mats, rho, p_decay, ps, Ntot, max_iter, tol ) 

% relative value iteration; see, e.g., Mahadevan (1996)
% 
% INPUTS
% tau_mats: transition function
% rho: immediate reward function 
% p_decay: between-trial probability of decay STRONG->WEAK
% ps: beliefs
% Ntot: number of trial states (excluding initial and decision)
% max_iter: maximum number of iterations
% tol: minimum change in value function to accept convergence
% 
% OUTPUTS
% a_opt: optimal policy
% r_av: optimal average reward rate
% v: value function
% Q_att: q-function
% a_opt_ch: at decision state, whether report signal or not

nps=size(ps,1);
pstot = [ps; ps]; 
npstot=size(pstot,1);

v = zeros(npstot,Ntot+2); % relative value of occupying each state at each timestep; +2 because of initial state and final state where decision made
vnew = v;
Q_att = zeros(npstot,Ntot+2,2); % value of attention action (WEAK(1) or STRONG(2)) at each state, and at each timestep
Q_ch = zeros(npstot,2); % just applies to the choice of whether to report a signal or not
a_opt =  zeros(npstot,Ntot+2);
% final reward : here we assume always a binary reward of either 0 (if incorrect) or 1 (if correct) at the end
Q_ch(:,1) = pstot(:,1); % report 'no signal'
Q_ch(:,2) = (pstot(:,2)+pstot(:,3)); % report 'signal'
[V_ch,a_opt_ch] = max( [Q_ch(:,1) Q_ch(:,2)], [], 2 );

binit = [1 0 0]; % initially, complete belief that there is no signal present
dist = sum(abs(ps-binit),2);
[~,id_start_low] = min(dist);
id_start_high = nps + id_start_low;

iter = 0;
diff=100;
r_av = 0;
fprintf('performing value iteration....\n\n')
while iter<max_iter && diff>tol
    iter = iter+1;
    Q_att(:,end,1) = rho(:,1) + v(id_start_low,1); % choose WEAK at final step
    Q_att(:,end,2) = rho(:,2) + (1-p_decay)*v(id_start_high,1) + p_decay*v(id_start_low,1); % choose STRONG at final step
    [vnew(:,end), a_opt(:,end)] = max( [V_ch+Q_att(:,end,1), V_ch+Q_att(:,end,2)], [], 2 );
    for n = (Ntot+1):-1:1
        for i=1:npstot % for each current belief
            for k=1:2 % for each action (WEAK,STRONG)
                Q_att(i,n,k) = rho(i,k) + tau_mats{k}(i,:,n)*vnew(:,n+1);
            end
            [vnew(i,n), a_opt(i,n)] = max(Q_att(i,n,:));
        end
    end
    r_av(iter+1) = vnew(id_start_low)/(Ntot+2);
    if max_iter>1 % NB: only if we're solving for the average reward case
        vnew = vnew - vnew(id_start_low);
    end
    diff = max(max(abs(vnew-v)));
    v = vnew;
    fprintf('iter %d, r_av = %6.2f, diff = %6.2f\n', iter, r_av(iter+1), diff);
end