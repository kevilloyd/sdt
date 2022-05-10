% Solve simple signal detection problem for optimal attentional actions.
% (runs successfully with Matlab R2020b)

% main output is beliefs(Nsig+1,Nsig+1,Ntot+2,6,2), where:
% Nsig: number of time steps when signal can come on
% Ntot: total number of time steps, excluding first step and final decision step
% 6 = number of hidden experiment states(3) x number of possible attentional states(2)
% 2 = either starting a trial with 1) WEAK attention or 2)STRONG
% attention
%
% e.g. beliefs(start,stopper,t,1:3,1) is the case that the signal comes on at 'start'
% and off at 'stopper'; at step t, how much belief is there in the WEAK attention versions of
% states 1-3; this is for the case where the trial starts with WEAK attention (last argument=1).
%
% beliefs(start,stopper,t,4:6,1) is the same for the STRONG attention case,
% starting the trial with attention WEAK.
%
% Plot these as e.g., plotter(squeeze(beliefs(3,7,:,:,1)));

clear; close all

rng(0)

do_save = true;
do_plots = true;
compute_alternative_policies = true; % if you just want optimal policy, set this to 'false'

initialize; % call 'initialize.m' where most important parameters are set

save_name = strcat( 'p',num2str(p_sig),'_q',num2str(q),'_s0w',num2str(s0w),'_s1w',num2str(s1w), '_s0s', num2str(s0s),'_s1s', num2str(s1s) , '_du',num2str(c_du),...
    '_uu',num2str(c_uu),'_Pdecay',num2str(p_decay),'_Nsig',num2str(Nsig),'_Nadd',num2str(Nadd),'_pnum',num2str(pnum),'_interp',num2str(interp),'_samples',num2str(samples),...
    '_maxiter',num2str(max_iter),'_gauss','.mat' );

% we need to derive the transition matrix over beliefs, tau_mats (Equations 12,13 in paper);
% imats holds the corresponding indices of belief states and their weights;
% note that this is a function of time;
% we also save the (interpolated) consequences of observations for the
% averaging equations;
% maxd tells us the interpolation error
tau_mats = cell(2,1); % dependence on action (i.e., attention: 1=weak, 2=strong)
imats = cell(2,1);
maxd=zeros(2,1);
for att=1:2
    [tau_mats{att},imats{att},maxd(att)] = get_belief_transitions_momdp( ps, T, O{1,att}, O{2,att}, att, interp, p_decay );
end

% rewards (Equation 14 in paper)
rho = zeros(npstot,2);
rho(1:nps,2) = ps*R(1:3,2);
rho(nps+1:end,2) = ps*R(4:6,2); 

%% SOLVE FOR OPTIMAL POLICY (cf. Equations 15 and 16 in paper)
[a_opt, r_av, V, Q_att, a_opt_ch] = rel_value_iteration( tau_mats, rho, p_decay, ps, Ntot, max_iter, tol );

if compute_alternative_policies
    
    a_w = ones(size(a_opt)); % evaluation 'always weak' policy,
    [r_av_w, V_w] = rel_policy_evaluation_matched( a_w, tau_mats, rho, p_decay, ps, Ntot, max_iter, tol );
    
    a_s = ones(size(a_opt)).*2; % evaluation 'always strong' policy
    [r_av_s, V_s] = rel_policy_evaluation_matched( a_s, tau_mats, rho, p_decay, ps, Ntot, max_iter, tol );
    
end

%% COMPUTE TRAJECTORIES GIVEN POLICY(S)

% for optimal policy
[ beliefs, beliefs_samp, beliefs_samp0, beliefs_samp1, prs, prs_samp, Patton_samp ] = compute_trajectories(...
    a_opt, a_opt_ch, imats, T, O, ps, p_decay, Nadd, Nsig, samples, interp );

if compute_alternative_policies
    
    % for 'always weak', but with awareness of that fact
    [ beliefs_w, beliefs_samp_w, beliefs_samp0_w, beliefs_samp1_w, prs_w, prs_samp_w, Patton_samp_w ] = compute_trajectories(...
        a_w, a_opt_ch, imats, T, O, ps, p_decay, Nadd, Nsig, samples, interp );
    
    % for 'always strong', but with awareness of that fact
    [ beliefs_s, beliefs_samp_s, beliefs_samp0_s, beliefs_samp1_s, prs_s, prs_samp_s, Patton_samp_s ] = compute_trajectories(...
        a_s, a_opt_ch, imats, T, O, ps, p_decay, Nadd, Nsig, samples, interp );
    
    % for optimal policy, but with "ACh lesion": behave according to a_opt, but
    % can't actually reach strong attention -- but treat those observations as
    % if they were experienced in strong attentional state! (i.e., unaware of lesion)
    [ beliefs_l, beliefs_samp_l, beliefs_samp0_l, beliefs_samp1_l, prs_l, prs_samp_l, Patton_samp_l ] = compute_trajectories_lesion(...
        a_opt, a_opt_ch, imats, T, O, ps, p_decay, Nadd, Nsig, samples, interp );
    
end

%% POST-PROCESSING (additional quantities that might be useful to compute here rather than later)
% for optimal policy
[av_beliefs, av_beliefs_on, av_beliefs_off, av_beliefs_samp, av_beliefs_on_samp, av_beliefs_off_samp,...
    av_beliefs_on0_samp, av_beliefs_on1_samp, av_beliefs_off0_samp, av_beliefs_off1_samp,...
    dprime, dprime_av, F, H, dprime_overall, crit, dprime_samp, dprime_av_samp, F_samp, H_samp, dprime_overall_samp, crit_samp,...
    Patton_overall, Patton_off, Patton_on] = ...
    get_results( beliefs, beliefs_samp, beliefs_samp0, beliefs_samp1, ons, prs, prs_samp, Patton_samp, Nadd, Nsig  );

if compute_alternative_policies
    
    % for 'always weak', 'default' policy ("ACh lesion with awareness")
    [av_beliefs_w, av_beliefs_on_w, av_beliefs_off_w, av_beliefs_samp_w, av_beliefs_on_samp_w, av_beliefs_off_samp_w,...
        av_beliefs_on0_samp_w, av_beliefs_on1_samp_w, av_beliefs_off0_samp_w, av_beliefs_off1_samp_w,...
        dprime_w, dprime_av_w, F_w, H_w, dprime_overall_w, crit_w, dprime_samp_w, dprime_av_samp_w, F_samp_w, H_samp_w, dprime_overall_samp_w, crit_samp_w,...
        Patton_overall_w, Patton_off_w, Patton_on_w] = ...
        get_results( beliefs_w, beliefs_samp_w, beliefs_samp0_w, beliefs_samp1_w, ons, prs_w, prs_samp_w, Patton_samp_w, Nadd, Nsig  );
    
    % for 'always strong'
    [av_beliefs_s, av_beliefs_on_s, av_beliefs_off_s, av_beliefs_samp_s, av_beliefs_on_samp_s, av_beliefs_off_samp_s,...
        av_beliefs_on0_samp_s, av_beliefs_on1_samp_s, av_beliefs_off0_samp_s, av_beliefs_off1_samp_s,...
        dprime_s, dprime_av_s, F_s, H_s, dprime_overall_s, crit_s, dprime_samp_s, dprime_av_samp_s, F_samp_s, H_samp_s, dprime_overall_samp_s, crit_samp_s,...
        Patton_overall_s, Patton_off_s, Patton_on_s] = ...
        get_results( beliefs_s, beliefs_samp_s, beliefs_samp0_s, beliefs_samp1_s, ons, prs_s, prs_samp_s, Patton_samp_s, Nadd, Nsig  );
    
    % for "ACh lesion without awareness"
    [av_beliefs_l, av_beliefs_on_l, av_beliefs_off_l, av_beliefs_samp_l, av_beliefs_on_samp_l, av_beliefs_off_samp_l,...
        av_beliefs_on0_samp_l, av_beliefs_on1_samp_l, av_beliefs_off0_samp_l, av_beliefs_off1_samp_l,...
        dprime_l, dprime_av_l, F_l, H_l, dprime_overall_l, crit_l, dprime_samp_l, dprime_av_samp_l, F_samp_l, H_samp_l, dprime_overall_samp_l, crit_samp_l,...
        Patton_overall_l, Patton_off_l, Patton_on_l] = ...
        get_results( beliefs_l, beliefs_samp_l, beliefs_samp0_l, beliefs_samp1_l, ons, prs_l, prs_samp_l, Patton_samp_l, Nadd, Nsig  );
    
end

%% SAVE?
if do_save
    save(save_name,'-v7.3')
end

%% FIGURES?
if do_plots
    plot_values( V, Q_att, a_opt, ps, pnum )
    
    plot_trajectories( beliefs, beliefs_samp, av_beliefs_on, av_beliefs_on_samp, av_beliefs_off, av_beliefs_off_samp,...
        av_beliefs_on0_samp, av_beliefs_on1_samp, av_beliefs_off0_samp, av_beliefs_off1_samp, Nadd, q)
    
    plot_performance( prs, prs_samp, ons, Nadd, q )
end
