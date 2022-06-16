function [av_beliefs, av_beliefs_on, av_beliefs_off, av_beliefs_samp, av_beliefs_on_samp, av_beliefs_off_samp,...
    av_beliefs_on0_samp, av_beliefs_on1_samp, av_beliefs_off0_samp, av_beliefs_off1_samp,...
    dprime, dprime_av, F, H, dprime_overall, crit, dprime_samp, dprime_av_samp, F_samp, H_samp, dprime_overall_samp, crit_samp,...
    Patton_overall, Patton_off, Patton_on] = ...
                                get_results( beliefs, beliefs_samp, beliefs_samp0, beliefs_samp1, ons, prs, prs_samp, Patton_samp, Nadd, Nsig  )

% some additional post-processing to extract further quantities of interest

% INPUTS
% beliefs: exact average belief trajectories by each possible onset and offset
% beliefs_samp: sampled equivalent of previous
% beliefs_samp0: average (from samples) belief trajectories for all cases when responded 'no signal'
% beliefs_samp1: average (from samples) belief trajectories for all cases when responded 'signal'
% ons: probabilities for all onset-offset combinations (derived from p_sig and hazard)
% prs: exact probabilities of reporting 'signal' in each case
% prs_samp: sampled equiv. of previous
% Patton_samp: probability of having strong attention in a trial (estimated from sampling) 
% Nadd: number of initial non-signal states (in addition to the very first time step)
% Nsig: number of signal states
% 
% OUTPUTS 
% av_beliefs: average overall belief trajectory
% av_beliefs_on: average belief trajectory conditional on signal ON
% av_beliefs_off: "                  "                "          OFF
% av_beliefs_samp: sampling versions of above
% av_beliefs_on_samp: "                      "
% av_beliefs_off_samp: "                     "
% av_beliefs_on0_samp:  conditioned on both signal ON and report  'NO SIGNAL'
% av_beliefs_on1_samp:  "                        " ON "        "  'SIGNAL'
% av_beliefs_off0_samp: "                        " OFF "        " 'NO SIGNAL'
% av_beliefs_off1_samp: "                        " OFF "        " 'SIGNAL'
% dprime: sensitivities for all possible times on and off
% dprime_av: sensitivities averaging over previous
% F: false alarm rate
% H: hit rate
% dprime_overall: overall sensitivity
% crit: criterion
% dprime_samp:         sampled versions of above
% dprime_av_samp:      "                       "
% F_samp:              "                       "
% H_samp:              "                       "
% dprime_overall_samp: "                       "
% crit_samp:           "                       "
% Patton_overall: overall probabilities of strong given initial attentional state
% Patton_off:     "                        "                   "                " and signal OFF
% Patton_on:      "                        "                   "                " and signal ON

% for when conditioning on cases where signal on
weights_on = ons(Nadd+1:end-1,Nadd+1:end)/(1-ons(end,end));

%% PSYCHOPHYS (d', etc.)
% all dprime
dprime = norminv(prs(1:end-1,:,:)) - norminv(prs(end,end,:));
dprime_samp = norminv(prs_samp(1:end-1,:,:)) - norminv(prs_samp(end,end,:));
dprime(isinf(dprime)) = nan;
dprime(dprime<0) = 0;
dprime_samp(isinf(dprime_samp)) = nan;
dprime_samp(dprime_samp<0) = 0;
if sum(dprime(:,end) - dprime(:,end-1))>1e-32
    error('last two columns should have same d-prime!!')
end
dprime = dprime(:,1:end-1,:); % so this d' for all cases (i.e., for each possible onset and offset)
dprime_samp(:,end-1,:) = mean( dprime_samp(:,end-1:end,:), 2, 'omitnan' );
dprime_samp = dprime_samp(:,1:end-1,:);
F = squeeze( prs(end,end,:) ); % false alarm rates (starting weak and strong)
H = weights_on.*prs(1:end-1,:,:); 
H = squeeze(sum(sum(H))); % overall hit rates (starting weak and strong)
F_samp = squeeze( prs_samp(end,end,:) );
H_samp = weights_on.*prs_samp(1:end-1,:,:);
H_samp = squeeze(sum(sum(H_samp)));
dprime_overall = norminv(H) - norminv(F); % overall d'
dprime_overall_samp = norminv(H_samp) - norminv(F_samp);
crit = -.5*(norminv(H)+norminv(F)); % overall decision criterion
crit_samp = -.5*(norminv(H_samp)+norminv(F_samp));
dprime_av = zeros(Nsig,2); % for each possible signal length, for each initial attentional state, averaging over different onsets
dprime_av_samp = zeros(Nsig,2);
weights_on_temp = weights_on;
weights_on_temp(:,end-1) = sum(weights_on(:,(end-1):end),2);
weights_on_temp = weights_on_temp(:,1:end-1);
for att=1:2
    for i=1:Nsig
        weights = diag(weights_on_temp,i-1);
        weights = weights./sum(weights);
        dprime_av(i,att) = sum( weights.*diag(dprime(:,:,att),i-1) );
        dprime_av_samp(i,att) = sum( weights.*diag(dprime_samp(:,:,att),i-1) );
    end
end

%% WORK OUT ALL EXPECTED TRAJECTORIES OF INTEREST
ons_temp= ons(1+Nadd:end,1+Nadd:end);
av_beliefs = repmat(ons_temp,[1,1,1+Nadd+Nsig+2,6,2]).*beliefs;
av_beliefs = squeeze( sum( av_beliefs, [1 2] ) ); % this is the OVERALL AVERAGE trajectory (i.e., unconditional)
av_beliefs_samp = repmat(ons_temp,[1,1,1+Nadd+Nsig+2,6,2]).*beliefs_samp;
av_beliefs_samp = squeeze( sum( av_beliefs_samp, [1 2] ) );
%% i.e., by SIGNAL (off vs. on)
%% i.e., by SIGNAL x RESPONSE (report 'no signal' vs. 'signal')
weights_on1 = repmat(weights_on,[1 1 2]).*prs(1:end-1,:,:); % signal on, report on
weights_on1(:,:,1) = weights_on1(:,:,1)./sum(sum(weights_on1(:,:,1)));
weights_on1(:,:,2) = weights_on1(:,:,2)./sum(sum(weights_on1(:,:,2)));
weights_on0 = repmat(weights_on,[1 1 2]).*(1-prs(1:end-1,:,:)); % signal on, report off
weights_on0(:,:,1) = weights_on0(:,:,1)./sum(sum(weights_on0(:,:,1)));
weights_on0(:,:,2) = weights_on0(:,:,2)./sum(sum(weights_on0(:,:,2)));
%
beliefs_on = beliefs(1:Nsig,:,:,:,:);
beliefs_on_samp = beliefs_samp(1:Nsig,:,:,:,:);
beliefs_on0_samp = beliefs_samp0(1:Nsig,:,:,:,:);
beliefs_on1_samp = beliefs_samp1(1:Nsig,:,:,:,:);
wbeliefs_on = zeros(size(beliefs_on));
wbeliefs_on_samp = zeros(size(beliefs_on_samp));
wbeliefs_on0_samp = zeros(size(beliefs_on0_samp));
wbeliefs_on1_samp = zeros(size(beliefs_on1_samp));
for start = 1:Nsig
    for stopper = start:(Nsig+1)
        for att=1:2
            wbeliefs_on(start,stopper,:,:,att) = weights_on(start,stopper).*beliefs_on(start,stopper,:,:,att);
            wbeliefs_on_samp(start,stopper,:,:,att) = weights_on(start,stopper).*beliefs_on_samp(start,stopper,:,:,att);
            wbeliefs_on0_samp(start,stopper,:,:,att) = weights_on0(start,stopper,att).*beliefs_on0_samp(start,stopper,:,:,att);
            wbeliefs_on1_samp(start,stopper,:,:,att) = weights_on1(start,stopper,att).*beliefs_on1_samp(start,stopper,:,:,att);
        end
    end
end
av_beliefs_on = squeeze( sum( wbeliefs_on, [1 2] ) );
av_beliefs_on_samp = squeeze( sum( wbeliefs_on_samp, [1 2] ) );
av_beliefs_on0_samp = squeeze( sum( wbeliefs_on0_samp, [1 2], 'omitnan' ) );
av_beliefs_on1_samp = squeeze( sum( wbeliefs_on1_samp, [1 2], 'omitnan' ) );
% and when the signal is OFF?
av_beliefs_off = squeeze( beliefs(end,end,:,:,:) );
av_beliefs_off_samp = squeeze( beliefs_samp(end,end,:,:,:) );
av_beliefs_off0_samp = squeeze( beliefs_samp0(end,end,:,:,:) );
av_beliefs_off1_samp = squeeze( beliefs_samp1(end,end,:,:,:) );

%% finally, some stuff we can get once we have the average trajectories, regarding ATTENTION
Patton_overall = squeeze( sum( repmat(ons_temp,[1,1,2]).*Patton_samp, [1 2], 'omitnan' ) ); % overall probabilities of strong given initial attentional state
Patton_on = squeeze( sum( repmat(weights_on,[1,1,2]).*Patton_samp(1:end-1,:,:), [1 2], 'omitnan' ) );
Patton_off = squeeze( Patton_samp(end,end,:) );
