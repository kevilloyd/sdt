function [r_av, v] = rel_policy_evaluation_matched( a_opt, tau_mats, rho, p_decay, ps, Ntot, max_iter, tol ) 

% relative policy iteration: this assumes that the given policy, and the
% treatment of observations arising under the policy, are MATCHED
% 
% INPUTS
% a_opt: the policy of interest
% tau_mats: transition function
% rho: immediate reward function 
% p_decay:
% ps: beliefs
% Ntot: number of trial states (excluding initial and decision)
% max_iter:
% tol:
% 
% OUTPUTS
% r_av:
% v:

nps=size(ps,1);
pstot = [ps; ps]; 
npstot=size(pstot,1);

v = zeros(npstot,Ntot+2); % relative value of occupying each state at each timestep; +2 because of initial state and final state where decision made
vnew = v;
Q_att = zeros(npstot,Ntot+2,2); % value of attention action (WEAK(1) or STRONG(2)) at each state, and at each timestep
Q_ch = zeros(npstot,2); % just applies to the choice of whether to report a signal or not
% final reward : here we assume always 0,1 at the end
Q_ch(:,1) = pstot(:,1); % report 'no signal'
Q_ch(:,2) = (pstot(:,2)+pstot(:,3)); % report 'signal'
[V_ch,~] = max( [Q_ch(:,1) Q_ch(:,2)], [], 2 );

binit = [1 0 0]; % initially, complete belief that there is no signal present
dist = sum(abs(ps-binit),2);
[~,id_start_low] = min(dist);
id_start_high = nps + id_start_low;

iter = 0;
diff=100;
r_av = 0;
fprintf('\n\n performing policy evaluation....\n\n')
while iter<max_iter && diff>tol
    iter = iter+1;
    % final step
    for i=1:npstot % for each current belief
        aa = a_opt(i,end);
        if aa==1
            vnew(i,end) = rho(i,1) + v(id_start_low,1); % choose WEAK at final step
        else
            vnew(i,end) = rho(i,2) + (1-p_decay)*v(id_start_high,1) + p_decay*v(id_start_low,1); % choose STRONG at final step
        end
    end
    vnew(:,end) = V_ch + vnew(:,end);
    % for the rest
    for n = (Ntot+1):-1:1
        for i=1:npstot % for each current belief
                aa = a_opt(i,n);
                vnew(i,n) = rho(i,aa) + tau_mats{aa}(i,:,n)*vnew(:,n+1); % crucial step in whether to assume matched            
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