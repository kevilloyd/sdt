function [ beliefs, beliefs_samp, beliefs_samp0, beliefs_samp1, prs, prs_samp, Patton_samp ] = compute_trajectories(...
                                                                a_opt, a_opt_ch, imats, T, O, ps, p_decay, Nadd, Nsig, samples, interp )

% take a policy, and work out what the trajectories are going to be like (see Section 2.3 and Fig.4 of manuscript)

% INPUTS
% a_opt: the policy of interest
% a_opt_ch: decision policy (always the same)
% imats: transition function for beliefs
% T: transition function for underlying states
% O: observation function
% ps: all possible beliefs
% p_decay: probability of decaying strong -> weak between trials, even if chose strong (cf. lower table in Fig.1E)  
% Nadd: number N_0 of time steps where it is known that no signal can be present 
% Nsig: number N_1 of time steps during which a signal may arrive
% samples: number of samples for sampling solution
% interp: number of belief points to interpolate 

nStot = 6;
Ntot = Nadd+Nsig;
nps=size(ps,1);
pstot = [ps; ps]; 
npstot=size(pstot,1);
binit = [1 0 0]; % initially, complete belief that there is no signal present
dist = sum(abs(ps-binit),2);
[~,id_start_low] = min(dist);
id_start_high = nps + id_start_low;

psmunge=[ps' 0*ps'; 0*ps' ps']; % all possible states (states 1-3 for weak attention, states 4-6 for strong attention)

beliefs=zeros(Nsig+1,Nsig+1,Ntot+3,nStot,2); % exact; note now Ntot+3, because of inclusion of belief at START OF NEXT TRIAL
beliefs_samp = beliefs; % sampled
beliefs_samp0 = beliefs_samp; % sampled -- average beliefs when report 'no signal'
beliefs_samp1 = beliefs_samp; % sampled -- average beliefs when report 'signal'
prs = zeros(Nsig+1,Nsig+1,2); % for each possible start and stop time, and attentional state at the beginning of the trial, what is probability of reporting signal? 
prs_samp = prs; % from sampling
Patton_samp = nan(size(prs)); % for each possible..., what is the probability of ever being in the strong attentional state?
id_start = [id_start_low id_start_high];
fprintf('\n\n computing average trajectories...\n\n');
for start = 1:Nsig  
    fprintf('on %2d\n',start);
    for stopper = start:(Nsig+1) 
    % having separated out by when the signal comes on
      % we can work out the belief trajectory for all possibilities
      % and can average using the matrix ons if necessary
        trans = get_belief_momdp( imats, a_opt, O, start+Nadd+1, stopper+Nadd+1, p_decay );
        for att=1:2
            currentp=zeros(1,npstot);
            currentp(id_start(att))=1;
            for n=1:Ntot+1
                beliefs(start,stopper,n,:,att)=psmunge*currentp';
                currentp=currentp*trans(:,:,n);
            end
            beliefs(start,stopper,Ntot+2,:,att)=psmunge*currentp'; % final decision state!
            prs(start,stopper,att) = sum( currentp(a_opt_ch==2) ); % we know the beliefs from which signal would be reported!
            currentp=currentp*squeeze(trans(:,:,end)); % turn the crank one more time to get belief at START OF NEXT TRIAL
            beliefs(start,stopper,Ntot+3,:,att) = psmunge*currentp';
            % sample
            [beliefs_samp(start,stopper,:,:,att),beliefs_samp0(start,stopper,:,:,att),beliefs_samp1(start,stopper,:,:,att),~, Patton_samp(start,stopper,att), prs_samp(start,stopper,att)] = ...
                samplerf(ps, T, O, a_opt, start+Nadd+1, stopper+Nadd+1, samples, interp, att, p_decay); 
        end
    end
end
trans = get_belief_momdp( imats, a_opt, O, Ntot+2, Ntot+2, p_decay); % for NON-SIGNAL trial
for att=1:2
    currentp=zeros(1,npstot);
    currentp(id_start(att))=1;
    for n=1:Ntot+1
        beliefs(Nsig+1,Nsig+1,n,:,att)=psmunge*currentp';
        currentp=currentp*trans(:,:,n);
    end
    beliefs(Nsig+1,Nsig+1,Ntot+2,:,att)=psmunge*currentp'; % final decision state!
    prs(Nsig+1,Nsig+1,att) = sum( currentp(a_opt_ch==2) );
    currentp=currentp*squeeze(trans(:,:,end)); % turn the crank one more time to get belief at START OF NEXT TRIAL
    beliefs(Nsig+1,Nsig+1,Ntot+3,:,att) = psmunge*currentp';
    [beliefs_samp(Nsig+1,Nsig+1,:,:,att), beliefs_samp0(Nsig+1,Nsig+1,:,:,att), beliefs_samp1(Nsig+1,Nsig+1,:,:,att), ~, Patton_samp(Nsig+1,Nsig+1,att), prs_samp(Nsig+1,Nsig+1,att)] = ...
        samplerf(ps, T, O, a_opt, Nadd+Nsig+2, Nadd+Nsig+2, samples, interp, att, p_decay);
end