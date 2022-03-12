function [nicebel,nicebel0,nicebel1,atts,p_atton,prs]=samplerf_v2_lesion_gauss(ps,T,O,a_opt,start,stopper,samples,interp,att_start,p_decay)

% many sample belief trajectories given optimal policy
%
% INPUTS
% ps: all possible beliefs
% T: transition matrix
% O: observation matrix
% a_opt: optimal policy
% start: fixed time step for signal to turn ON
% stopper: fixed time step for signal to turn OFF
% samples: number of samples
% interp: number of belief points to interpolate
% att_start: initial attentional state at start of trial (1=WEAK, 2=STRONG)
% p_decay: probability of decay to WEAK at start of next trial, even if
%               chose STRONG at decision state of previous trial
% lesion: if true, then all observations are generated as if always in weak
% attentional state, though not necessarily treated as such!
%
% OUTPUTS
% nicebel: average belief trajectory overall
% nicebel0: average belief trajectory for report 'no signal' 
% nicebel1: average belief trajectory for report 'signal'
% atts: average occupancy of STRONG state at each time step
% p_atton: probability of strong attention being engaged at all
% prs: proportion/probability of reporting 'signal' 

nSe=size(T,1);
N=size(T,3); 
nps=size(ps,1);

psmunge=[ps' 0*ps'; 0*ps' ps'];

nicebel = zeros(N+1,6,samples);
nicebel1=zeros(N+1,6);
nicebel0=zeros(N+1,6);

onoff=zeros(1,N);
if (start<=N)
    onoff(start:min(N,stopper))=1;
end

atts=zeros(samples,N+1);

response = zeros(samples,1); % for each sample, 0=report no signal, 1=report signal
for trial=1:samples
    currentp=[1 0 0];
    att=att_start;
    for t=1:N-1 % for all but the decision state! 
        fit=fitter(currentp,ps,interp);
        currentp=fit(:,2)'*ps(fit(:,1),:);
        nicebel(t,(att-1)*nSe+(1:nSe),trial) = currentp;
        atts(trial,t) = (att-1);
        att = 1 + ( (1+rand(1)) < (fit(:,2)'*a_opt(nps*(att-1)+fit(:,1),t)) ); % sample attention
        obser = 1 + sum( rand(1) > cumsum(O{onoff(t)+1,1}) );  % NB: if lesion, ALWAYS generated as if occupy weak state
        lik = [O{1,att}(obser) O{2,att}(obser) O{1,att}(obser)]; % ...BUT always treated as if observation were consistent with chosen attentional act!
        currentp = (currentp.*lik);
        currentp = currentp/sum(currentp);
        currentp = currentp*T(:,:,t);
    end
    fit=fitter(currentp,ps,interp);
    currentp=fit(:,2)'*ps(fit(:,1),:);
    nicebel(N,(att-1)*nSe+(1:nSe),trial) = currentp; % decision state!
    atts(trial,N) = (att-1);
    response(trial) = currentp(1) < .5; % if belief in 'presignal' less than half, report 'signal'!
    att = 1 + ( (1+rand(1)) < (fit(:,2)'*a_opt(nps*(att-1)+fit(:,1),N)) ); % sample attention in final, decision state
    atts(trial,N+1) = (att-1);
    currentp = currentp*T(:,:,N); % to get START OF NEXT TRIAL
    % but if choose strong attention in decision state, there's some
    % probability of decay!
    if att==2
       att = 1 + (rand(1)>p_decay); 
    end
    nicebel(N+1,(att-1)*nSe+(1:nSe),trial) = currentp;
    if response(trial)
        nicebel1 = nicebel1 + nicebel(:,:,trial); % adding up beliefs when reported signal
    else
        nicebel0 = nicebel0 + nicebel(:,:,trial); % adding up beliefs when reported no signal
    end
end
nicebel = mean(nicebel,3);
nicebel1 = nicebel1/sum(response==1);
nicebel0 = nicebel0/sum(response==0);
atts = atts(2:end,:); % the very first column of atts is actually just the initial state (before choice of attention on first time step)
att_on = sum(atts,2)>0; 
p_atton = mean(att_on); % so for a single trial, this is probability of attention switching on (always start with off); not necessarily true of continuing case
atts=mean(atts,1);
prs = mean(response); % proportion/probability report signal


