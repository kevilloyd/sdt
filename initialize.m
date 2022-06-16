% initialize the most important parameters here

Nsig = 6; % number of time steps per trial where the signal can come on (N1)
Nadd = 2; % number of ADDITIONAL time steps (i.e.,  in  addition to very first step) where the signal is KNOWN TO NOT ARRIVE (N0 - 1)
Ntot = Nadd+Nsig; % note that this does NOT include the initial state (where certain that signal not on) or final decision state (where choose to report signal or not)

p_sig = .5; % signal probability
q = .2;     % probability per time step that the signal turns OFF if it's ON
c_du = 1e-1;    % cost weak -> strong (kappa, in paper)
c_uu = 14e-3;   % cost strong -> strong (epsilon, in paper)
mus=[0 1];  % means (signal OFF vs. ON)
s0w =1;     % sds (w=WEAK, s=STRONG)
s0s = .5;
s1w = 1;
s1s = .5;
p_decay = 0.001; % assume that before starting next trial, there is some probability of 'decaying' to WEAK attentional state, even if chose STRONG

stds=[s0w s0s; s1w s1s]; % stds for off and on states (rows), without and with attn. (columns) 

max_iter = 100; % maximum iterations for value iteration; set this to 1 to solve for single-trial case
pnum=20;        % sets granularity of belief space by specifying a number of points on [0,1]
quarts=50;      % number of quantiles for observations x (since the observations continuous)
samples=2000;   % number of sample trajectories (to compare against exact solution)
interp = 3;     % number of belief points to interpolate
tol = 1e-6;     % close enough for convergence

nSi = 2;            % number of internal/observable states (i.e., whether attention weak or strong)
nSe = 3;            % number of external/unobservable states (i.e., whether 1=pre-signal, 2=signal, 3=post-signal)
nStot = nSi*nSe;    % therefore total number of states 
nA = 2;             % 2 possible actions: 1=WEAK, 2=STRONG (i.e., attentional action)

%% POMDP
% transition function T on the partially observed states (cf. tables in Fig.1C,D)
haz = [ zeros(1,Nadd) (p_sig/Nsig) ./ (1-p_sig.*((0:Nsig-1)./Nsig) ) ]; % i.e., hazard*dt (Equation 2 in paper) 
T = zeros(nSe,nSe,Ntot); % NB: applied at END of time step t
T(1,2,:) = haz;
T(1,1,:) = 1-T(1,2,:);
T(2,3,:) = q;
T(2,2,:) = 1-q;
T(3,3,:) = 1;
T(:,:,1:Nadd) = repmat(eye(nSe),[1 1 Nadd]);    % additional non-signal states (if any)
T = cat(3,T,[1 0 0; 0 0 1; 0 0 1]);             % transition into decision state
Tlast = [1 0 0; 1 0 0; 1 0 0]; % i.e., consider also the transition from the decision state to the NEXT trial
T = cat(3,T,Tlast);
% reward function R
R = zeros(nStot,nA);
R(:,1) = 0; % assume no cost of choosing WEAK
R(1:3,2) = -c_du; % cost of going WEAK -> STRONG
R(4:6,2) = -c_uu; % cost of going STRONG -> STRONG
% observation function O; 
% note that once you've chosen action, you know if
% you're in the weak or strong attentional state (i.e., that component of state is observable)
qs=(1:(quarts-1))/quarts; % the quantiles
x=[];
for on=1:2
    for att=1:2
        newx = norminv(qs,mus(on),stds(on,att)); % so these are all the possible observations x, conditioned on state and action
        x=[x newx];
    end
end
% now O(s,a,o)=Pr(o|s,a); for all combinations (signal off or on, attention low or high), and for all possible x
O=cell(2,2);
for att=1:2
    for on=1:2
        ps=normpdf(x,mus(on),stds(on,att)); 
        O{on,att} = ps./sum(ps);
    end
end

% (belief) states for the unobserved states 
ps=[]; % each possible belief state, b
for i=1:pnum
    s1=(i-1)/(pnum-1);
    for j=1:(pnum+1-i)
        s2=(j-1)/(pnum-1);
        ps=[ps ; [s1 s2 1-s1-s2]]; 
    end
end
nps=size(ps,1);
pstot = [ps; ps]; % with the internal state, we now have double the number of states (i.e., beliefs when in weak and strong attentional states)
npstot=size(pstot,1);

% the probability of each signal-on step (row) and signal-off step
% (column). Note its size Ntot+1 x Ntot+1; the (Ntot+1,Ntot+1) case is where no signal
% is presented at all. (Ntot,Ntot+1) is where the signal came on at last time
% step and stayed on; (Ntot,Ntot) is where signal came on at last time step and
% signal turned off at end of this same step.
ons=zeros(Ntot+1,Ntot+1);
noton=(1-haz);
cnot=[1 cumprod(noton)];
onher=haz.*cnot(1:(end-1)); % the signal can come on at any time
afteroff=q*((1-q).^(0:(Nsig-1))); % once the signal comes on, there is constant probability q that it turns off
ons(Nadd+1,Nadd+1)=1-sum(onher);
for start=Nadd+1:Ntot
    ons(start,start+(0:(Ntot-start)))=afteroff(1:(Ntot-start+1));
    ons(start,Ntot+1)=1-sum(ons(start,:));
    ons(start,:)=ons(start,:)*onher(start);
end
ons(Ntot+1,Ntot+1)=1-sum(onher); 
