function [ftraj_w,ftraj_s,ftraj_av] = plot_trajectories( beliefs, beliefs_samp, av_beliefs_on, av_beliefs_on_samp, av_beliefs_off, av_beliefs_off_samp,...
    av_beliefs_on0_samp, av_beliefs_on1_samp, av_beliefs_off0_samp, av_beliefs_off1_samp, Nadd, q)

% plot average trajectories

default_font_size = 12;
default_axlabels_size = 12;
default_title_size = 12;

[Nsig,~,Ntot,~,~] = size(beliefs);
Nsig = Nsig-1;
Ntot = Ntot-1; % in this case, we've included belief at START OF NEXT TRIAL

%% average belief trajectories ALIGNED TO TRIAL START, separated by signal ON/OFF and start trial WEAK/STRONG
ftraj_av = figure();
htraj_av = tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:4
    ax = nexttile;    
    tstep = i;
    if i==1
        bel = av_beliefs_off(:,:,1);
        plotter( bel, ax )
        t = title('signal OFF, start trial WEAK');
    elseif i==2
        bel = av_beliefs_on(:,:,1);
        plotter( bel, ax )
        t = title('signal ON, start trial WEAK');
    elseif i==3
        bel = av_beliefs_off(:,:,2);
        plotter( bel, ax )
        t = title('signal OFF, start trial STRONG');
    else
        bel = av_beliefs_on(:,:,2);
        plotter( bel, ax )
        t = title('signal ON, start trial STRONG');
    end
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 Ntot],'xtick',5:5:20,'box','off')
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i==1
       leg = legend('pre','on','post','att','Location','West');
       set(leg,'FontSize',default_axlabels_size,'FontName','Arial')
    end
    if i>2
        xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')
    end  
end


%% just starting in the WEAK state?
ftraj_av = figure();
htraj_av = tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% 0.0174    0.4508    0.4672    0.4325
for i=1:2
    ax = nexttile;    
    tstep = i;
    if i==1
        bel = av_beliefs_off(:,:,1);
        plotter( bel, ax )
        t = title('signal OFF, start trial WEAK');
    elseif i==2
        bel = av_beliefs_on(:,:,1);
        plotter( bel, ax )
        t = title('signal ON, start trial WEAK');
    end
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 Ntot],'xtick',5:5:20,'box','off')
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i==1
       leg = legend('pre','on','post','att','Location','West');
       set(leg,'FontSize',default_axlabels_size,'FontName','Arial')
    end
    xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')  
end
%% location of subpart on figure
% xstart=.83;
% xend=.98;
% ystart=.75;
% yend=.95;
xstart=.53;
xend=.68;
ystart=.7;
yend=.9;
        axi = axes('position',[xstart ystart xend-xstart yend-ystart ]);
        bel=av_beliefs_off(:,:,1); T=size(bel,1); att = sum(bel(:,4:6),2); plot(axi, 1:T-1, att(2:end),'k:','linewidth',3);
        hold on
        bel=av_beliefs_on(:,:,1); T=size(bel,1); att = sum(bel(:,4:6),2); ax=gca; plot(ax, 1:T-1, att(2:end),'k','linewidth',3);
leg=legend({'off','on'},'Location','West');
set(gca,'xticklabel',{},'yticklabel',{},'box','off')

%% SAMPLE average belief trajectories, ALIGNED TO TRIAL START, separated by signal ON/OFF, start trial WEAK/STRONG, and RESPONSE (report 'OFF','ON')
% START WEAK
ftraj_av_w = figure();
htraj_av_w = tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:4
    ax = nexttile;    
    tstep = i;
    if i==1
        bel = av_beliefs_off0_samp(:,:,1);
        plotter( bel, ax )
        t = title('signal OFF, report OFF (CR)');
    elseif i==2
        bel = av_beliefs_off1_samp(:,:,1);
        plotter( bel, ax )
        t = title('signal OFF, report ON (FA)');
    elseif i==3
        bel = av_beliefs_on0_samp(:,:,1);
        plotter( bel, ax )
        t = title('signal ON, report OFF (M)');
    else
        bel = av_beliefs_on1_samp(:,:,1);
        plotter( bel, ax )
        t = title('signal ON, report ON (H)');
    end
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 Ntot],'xtick',5:5:20)
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i==1
       leg = legend('pre','on','post','att','Location','West');
       set(leg,'FontSize',default_axlabels_size,'FontName','Arial')
    end
    if i>2
        xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')
    end  
end
sgtitle('trial-aligned SAMPLE averages, start WEAK')
% START STRONG
ftraj_av_s = figure();
htraj_av_s = tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:4
    ax = nexttile;    
    tstep = i;
    if i==1
        bel = av_beliefs_off0_samp(:,:,2);
        plotter( bel, ax )
        t = title('signal OFF, report OFF (CR)');
    elseif i==2
        bel = av_beliefs_off1_samp(:,:,2);
        plotter( bel, ax )
        t = title('signal OFF, report ON (FA)');
    elseif i==3
        bel = av_beliefs_on0_samp(:,:,2);
        plotter( bel, ax )
        t = title('signal ON, report OFF (M)');
    else
        bel = av_beliefs_on1_samp(:,:,2);
        plotter( bel, ax )
        t = title('signal ON, report ON (H)');
    end
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 Ntot],'xtick',5:5:20)
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i==1
       leg = legend('pre','on','post','att','Location','West');
       set(leg,'FontSize',default_axlabels_size,'FontName','Arial')
    end
    if i>2
        xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')
    end  
end
sgtitle('trial-aligned SAMPLE averages, start STRONG')

%% SANITY CHECKS (comparing exact versus sampled)
% comparison between average trajectories (signal ON) for start WEAK/STRONG
figure
ax = subplot(2,2,1);
plotter( av_beliefs_on(:,:,1), ax )
title('exact, start WEAK')
ax = subplot(2,2,2);
plotter( av_beliefs_on_samp(:,:,1), ax )
title('sampled, start WEAK')
ax = subplot(2,2,3);
plotter( av_beliefs_on(:,:,2), ax )
title('exact, start STRONG')
ax = subplot(2,2,4);
plotter( av_beliefs_on_samp(:,:,2), ax )
title('sampled, start STRONG')
sgtitle('average, SIGNAL trial')
% comparison between average trajectories (signal OFF) for start WEAK/STRONG
figure
ax = subplot(2,2,1);
plotter( av_beliefs_off(:,:,1), ax )
title('exact, start WEAK')
ax = subplot(2,2,2);
plotter( av_beliefs_off_samp(:,:,1), ax )
title('sampled, start WEAK')
ax = subplot(2,2,3);
plotter( av_beliefs_off(:,:,2), ax )
title('exact, start STRONG')
ax = subplot(2,2,4);
plotter( av_beliefs_off_samp(:,:,2), ax )
title('sampled, start STRONG')
sgtitle('average, NON-SIGNAL trial')

 %%
sig_length = 3; % which signal length are we interested in as an example?
num_plots = Nsig-sig_length+1;
% ncols = ceil( num_plots/4 );
% nrows = ceil(num_plots/ncols);
ncols = 4; nrows = 1;
% ncols = 6; nrows=1;
%
ftraj_w = figure;
htraj_w = tiledlayout(nrows,ncols, 'Padding', 'none', 'TileSpacing', 'compact'); 
% set(ftraj_w, 'position', [0.0539    0.5550    0.4201    0.3208]);
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots  
    ax = nexttile;    
    tstep = i;
    bel = squeeze(beliefs(i,i+(sig_length-1),:,:,1));
    plotter(bel, ax);
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 size(bel,1)-1],'xtick',5:5:20,'box','off')
    t = title(['on at n = ',num2str(i+Nadd+1)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i==1
       leg = legend('pre','on','post','att','Location','West');
       set(leg,'FontSize',default_axlabels_size,'FontName','Arial')
    end
    if i>=(num_plots-1)
        xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')
    end  
end
tt = title(htraj_w, 'start trial WEAK');
set(tt,'FontName','Arial','FontSize',default_title_size)
%
ftraj_s = figure();
htraj_s = tiledlayout(nrows,ncols, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots  
    ax = nexttile;    
    tstep = i;
    bel = squeeze(beliefs(i,i+(sig_length-1),:,:,2));
    plotter(bel, ax);
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 size(bel,1)],'xtick',5:5:20)
    t = title(['on at n = ',num2str(i+Nadd+1)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i==1
       leg = legend('pre','on','post','att','Location','West');
       set(leg,'FontSize',default_axlabels_size,'FontName','Arial')
    end
    if i>=(num_plots-1)
        xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')
    end  
end
tt = title(htraj_s, 'start trial STRONG');
set(tt,'FontName','Arial','FontSize',default_title_size)

%% put the previous 2 plots into a single plot?
% close all
sig_length = 3; % which signal length are we interested in as an example?
num_plots = (Nsig-sig_length+1)*2;
nrows = ceil( num_plots/4 );
ncols = ceil(num_plots/nrows);
%
ftraj_w = figure();
htraj_w = tiledlayout(nrows,ncols, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots/2
    % start weak
    ax = nexttile(i);
    tstep = i;
    bel = squeeze(beliefs(i,i+(sig_length-1),:,:,1));
    plotter(bel, ax);
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 size(bel,1)-1],'xtick',5:5:20,'box','off')
    t = title(['on at n = ',num2str(i+Nadd+1)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i==1
       leg = legend('pre','on','post','att','Location','West');
       set(leg,'FontSize',default_axlabels_size,'FontName','Arial')
    end
    if i>=(num_plots-1)
        xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')
    end  
    % start strong
    ax = nexttile(ncols+i);   
    tstep = i;
    bel = squeeze(beliefs(i,i+(sig_length-1),:,:,2));
    plotter(bel, ax);
    set(ax,'ylim',[0 1],'ytick',[0 0.5 1],'xlim',[1 size(bel,1)-1],'xtick',5:5:20,'box','off')
    t = title(['on at n = ',num2str(i+Nadd+1)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
    if i>=(num_plots-1)
        xlabel(ax, 'time step', 'FontSize', default_axlabels_size,'FontName','Arial')
    end  
end
% set(tt,'FontName','Arial','FontSize',default_title_size)


end
