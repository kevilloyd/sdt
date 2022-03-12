function [fhAw, fhAs, fhVw, fhQdiffw, fhQdiffs] = plot_values( V, Q, a_opt, ps, pnum ) 

% plot value functions and associated optimal policy

default_font_size = 8;
default_axlabels_size = 10;
default_title_size = 12;
default_width = 5.2;
default_height = 6;

[npstot,N] = size(V);
nps=npstot/2;

Qdiff_weak = Q(1:nps,:,2) - Q(1:nps,:,1);
Qdiff_strong = Q((nps+1):end,:,2) - Q((nps+1):end,:,1);
as_opt_weak = a_opt(1:nps,:);
as_opt_strong = a_opt((nps+1:end),:);

Vmax = max(max(V));
Vmin = min(min(V));
Qdiff_down_max = max(max(Qdiff_weak));
Qdiff_down_min = min(min(Qdiff_weak));
Qdiff_up_max = max(max(Qdiff_strong));
Qdiff_up_min = min(min(Qdiff_strong));

% for visualization, let's sort value functions, etc.; it's sufficient to
% represent the belief state with 2 numbers (we'll use b2 and b3) at any
% point in time
bs = linspace(0,1,pnum); % all possible belief values
nbs = length(bs);
idx = nan(nbs);
for i=1:nbs
    for j=1:nbs
       id = find( abs(ps(:,2)-bs(i))<1e-6 & abs(ps(:,3)-bs(j))<1e-6 ); % for possible belief [1-bs2-bs3, bs2=bs(i), bs3=bs(j)], what is its index? No index if impossible 
       if isempty(id)==0
           idx(i,j) = id;
       end
    end
end
idx_valid = find(isnan(idx)==0); % indices of idx that contain valid beliefs
Vv_weak = cell(N,1);
Vv_strong = cell(N,1);
Qqdiff_weak = cell(N,1);
Qqdiff_strong = cell(N,1);
ass_opt_weak = cell(N,1);
ass_opt_strong = cell(N,1);
for n=1:N
    Vv_weak{n} = nan(nbs);
    Vv_strong{n} = nan(nbs);
    Qqdiff_weak{n} = nan(nbs);
    Qqdiff_strong{n} = nan(nbs);
    ass_opt_weak{n} = nan(nbs);
    ass_opt_strong{n} = nan(nbs);
    %
    Vv_weak{n}(idx_valid) = V( idx(idx_valid), n );
    Vv_strong{n}(idx_valid) = V( nps + idx(idx_valid), n );
    Qqdiff_weak{n}(idx_valid) = Qdiff_weak( idx(idx_valid), n );
    Qqdiff_strong{n}(idx_valid) = Qdiff_strong( idx(idx_valid), n );
    ass_opt_weak{n}(idx_valid) = as_opt_weak( idx(idx_valid), n );
    ass_opt_strong{n}(idx_valid) = as_opt_strong( idx(idx_valid), n );
end

tdiff=1; % every tdiff timesteps
num_plots = ceil(N/tdiff);
num_rows = ceil(num_plots/5);
num_cols = ceil(num_plots/num_rows);

fhVw = figure();
hVw = tiledlayout(num_rows,num_cols, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots  
    nexttile    
    tstep = i;
    imagesc( [0 1], [0 1], Vv_weak{tstep}(:,:,1), [Vmin-.005,Vmax] );
    set(gca,'xtick',[0:.5:1],'ytick',[0:.5:1]);
    cmap = flipud(gray(256));
%     cmap = cmap([1 5:end],:);
    colormap(cmap);
%         colormap( [1 1 1; parula(256)] )
    axis square
    cbar = colorbar;
%     set(cbar,'Limits',[0.5 1],'Ticks',0.5:.1:1)
    xlabel('P(post-signal)', 'FontSize', default_axlabels_size,'FontName','Arial')
    ylabel('P(signal on)', 'FontSize', default_axlabels_size,'FontName','Arial')
    set(get(cbar,'title'),'string','V*')
    set(cbar,'fontSize',default_font_size,'FontName','Arial')
    t = title(['n = ' num2str(tstep)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
end
tt = title(hVw, 'state values, current attention weak');
set(tt,'FontName','Arial','FontSize',default_title_size)
subtitle(hVw,' ')
    
fhQdiffw = figure();
hQdiffw = tiledlayout(num_rows,num_cols, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots  
    nexttile    
    tstep = i;
    imagesc( [0 1], [0 1], Qqdiff_weak{tstep}, [Qdiff_down_min-.005,Qdiff_down_max] );
    set(gca,'xtick',[0:.5:1],'ytick',[0:.5:1]);
    cmap = flipud(gray(256));
%     cmap = cmap([1 5:end],:);
    colormap(cmap);
%         colormap( [1 1 1; parula(256)] )
    axis square
    cbar = colorbar;
    xlabel('P(post-signal)', 'FontSize', default_axlabels_size,'FontName','Arial')
    ylabel('P(signal on)', 'FontSize', default_axlabels_size,'FontName','Arial')
    set(get(cbar,'title'),'string','\Delta q*')
    set(cbar,'fontSize',default_font_size,'FontName','Arial')
    t = title(['n = ' num2str(tstep)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
end
tt = title(hQdiffw, 'q^*(strong)-q^*(weak), current attention weak');
set(tt,'FontName','Arial','FontSize',default_title_size)
subtitle(hQdiffw,' ')

fhQdiffs = figure();
hQdiffs = tiledlayout(num_rows,num_cols, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots  
    nexttile    
    tstep = i;
    imagesc( [0 1], [0 1], Qqdiff_strong{tstep}, [Qdiff_up_min-.005,Qdiff_up_max] );
    set(gca,'xtick',[0:.5:1],'ytick',[0:.5:1]);
%     cmap = flipud(gray(256));
%     cmap = cmap([1 5:end],:);
%     colormap(cmap);
        colormap( [1 1 1; parula(256)] )
    axis square
    cbar = colorbar;
    xlabel('P(post-signal)', 'FontSize', default_axlabels_size,'FontName','Arial')
    ylabel('P(signal on)', 'FontSize', default_axlabels_size,'FontName','Arial')
    set(get(cbar,'title'),'string','V*')
    set(cbar,'fontSize',default_font_size,'FontName','Arial')
    t = title(['n = ' num2str(tstep)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
end
tt = title(hQdiffs, 'q^*(strong)-q^*(weak), current attention strong');
set(tt,'FontName','Arial','FontSize',default_title_size)
subtitle(hQdiffs,' ')

fhAw = figure();
hAw = tiledlayout(num_rows,num_cols, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots  
    nexttile    
    tstep = i;
    imagesc( [0 1], [0 1], ass_opt_weak{tstep}, [1-1 2] );
    set(gca,'xtick',[0:.5:1],'ytick',[0:.5:1]);
    colormap( [1 1 1; .8 .8 .8; .4 .4 .4] )
    axis square
    cbar = colorbar;
    cbar.Ticks = [1.15 1.5];
    cbar.TickLabels = {'w','s'};
    cbar.Limits = [1 1.65];
    xlabel('P(post-signal)', 'FontSize', default_axlabels_size,'FontName','Arial')
    ylabel('P(signal on)', 'FontSize', default_axlabels_size,'FontName','Arial')
    set(get(cbar,'title'),'string','a*')
    set(cbar,'fontSize',default_font_size,'FontName','Arial')
    t = title(['n = ' num2str(tstep)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
end
tt = title(hAw, 'optimal actions, current attention weak');
set(tt,'FontName','Arial','FontSize',default_title_size)
subtitle(hAw,' ')

fhAs = figure();
hAs = tiledlayout(num_rows,num_cols, 'Padding', 'none', 'TileSpacing', 'compact'); 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
for i=1:num_plots  
    nexttile    
    tstep = i;
    imagesc( [0 1], [0 1], ass_opt_strong{tstep}, [1-1 2] );
    set(gca,'xtick',[0:.5:1],'ytick',[0:.5:1]);
    colormap( [1 1 1; .8 .8 .8; .4 .4 .4] )
    axis square
    cbar = colorbar;
    cbar.Ticks = [1.15 1.5];
    cbar.TickLabels = {'w','s'};
    cbar.Limits = [1 1.65];
    xlabel('P(post-signal)', 'FontSize', default_axlabels_size,'FontName','Arial')
    ylabel('P(signal on)', 'FontSize', default_axlabels_size,'FontName','Arial')
    set(get(cbar,'title'),'string','a*')
    set(cbar,'fontSize',default_font_size,'FontName','Arial')
    t = title(['n = ' num2str(tstep)]);
    set(t,'fontSize',default_font_size,'FontName','Arial')
end
tt = title(hAs, 'optimal actions, current attention strong');
set(tt,'FontName','Arial','FontSize',default_title_size)
subtitle(hAs,' ')

end