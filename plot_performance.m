function [fdprime, fdprime_ov] = plot_performance( prs, ~, ons, Nadd, q )

% here we want to look at d-prime

N = size(prs,1)-1;

weights_on = ons(Nadd+1:end-1,Nadd+1:end)/(1-ons(end,end));
weights_on(:,end-1) = sum(weights_on(:,(end-1):end),2);
weights_on = weights_on(:,1:end-1);

dprime = norminv(prs(1:end-1,:,:)) - norminv(prs(end,end,:));
dprime(dprime<0 | isinf(dprime)) = 0;
dprime_max = max(max(max(dprime)));
if sum(dprime(:,end) - dprime(:,end-1))>1e-32
    error('last two columns should have same d-prime!!')
end
dprime = dprime(:,1:end-1,:);
hs = prs(1:end-1,1:end-1,:);

dprime_overall = zeros(N,2); % for each possible signal length, for each initial attentional state, averaging by onset
F = squeeze( prs(end,end,:) ); % false alarm rates (starting weak and strong)
H = weights_on.*prs(1:end-1,1:end-1,:); 
H = squeeze(sum(sum(H))); % overall hit rates (starting weak and strong)
D = norminv(H) - norminv(F); % OVERALL d'
C = -.5*(norminv(H)+norminv(F));
fs = zeros(size(hs));
fs(:,:,1) = F(1);
fs(:,:,2) = F(2);
crit = -.5.*(norminv(hs)+norminv(fs));
if q<1e-3 % this means that once on, always stays on
    for att=1:2
        for i=1:N
            dprime_overall(i,att) = dprime(i,end,att);
        end
    end
else
    for att=1:2
        for i=1:N
            weights = diag(weights_on,i-1);
            weights = weights./sum(weights);
            dprime_overall(i,att) = sum( weights.*diag(dprime(:,:,att),i-1) );
            crit_overall(i,att) = sum( weights.*diag(crit(:,:,att),i-1) );
        end
    end
end

if q<1e-3 % since the signal always stays on once on, just look at performance as function of start time
    figure
    hold on
    plot((1+Nadd+1):(N+Nadd), dprime_overall(:,1),'k--o','markerSize',10,'MarkerFaceColor','k')
    plot((1+Nadd+1):(N+Nadd), dprime_overall(:,2),'k--v','markerSize',10,'MarkerFaceColor','k')
    leg = legend({'OFF','ON'},'Location','SouthEast');
    title(leg, 'start att.');
    xlabel('signal start')
    ylabel('d-prime')
    title('overall d-prime')
else
    figure
    subplot(1,2,1)
    imagesc( dprime(:,:,1), [0-.1 dprime_max] );
    colormap( [1 1 1; parula(256)] )
    set(gca,'xtick',1:1:10,'ytick',1:1:10,'xticklabel',Nadd+2:Nadd+N+1,'yticklabel',Nadd+2:Nadd+N+1)
    axis square
    cbar = colorbar;
    xlabel('signal off')
    ylabel('signal on)')
    title('start trial attention WEAK')
    subplot(1,2,2)
    imagesc( dprime(:,:,2), [0-.1 dprime_max] );
    colormap( [1 1 1; parula(256)] )
    set(gca,'xtick',1:1:10,'ytick',1:1:10,'xticklabel',Nadd+2:Nadd+N+1,'yticklabel',Nadd+2:Nadd+N+1)
    axis square
    cbar = colorbar;
    xlabel('signal off')
    ylabel('signal on)')
    title('start trial attention STRONG')
    sgtitle('d-prime')
    markers = {'o','v','d','^','s','>','<'};
    colours = lines(N);
    attstrs = {' WEAK',' STRONG'};
    
    fdprime = figure();
    ax = axes(fdprime);
    hold on
    for att = 1:2
        for i=1:N
            marker = markers{max(1,mod(i,length(markers)))};
            if att==1
                plot( ax, 1:1:(N-i+1), diag(dprime(:,:,att+1),i-1), 'color', colours(i,:), 'linestyle','--', 'marker', marker, 'markerSize', 10, 'MarkerFaceColor', colours(i,:) )
            else
                plot( ax, 1:1:(N-i+1), diag(dprime(:,:,att-1),i-1), 'color', colours(i,:), 'linestyle',':', 'marker', marker, 'markerSize', 10 )
            end
        end
        leg = legend(ax, {num2str([1:N]')}, 'Location', 'NorthEast');
        title(leg,'duration')
    end
    set(ax,'ylim',[0 4],'xlim',[.5 N+.5], 'xtick', [1:N], 'ytick',0:1:4, 'xticklabel',Nadd+2:Nadd+N+1)
    xlabel('signal onset')
    ylabel('d-prime') 
    %
    fdprime_ov = figure();
    hold on
    plot(flipud(dprime_overall(:,1)),'k--o','markerSize',10)
    plot(flipud(dprime_overall(:,2)),'k--o','markerSize',10,'MarkerFaceColor','k')
    leg = legend({'weak','strong'},'Location','NorthEast');
    title(leg, 'start att.');
    set(gca,'xlim',[.8 N+.2],'xtick',1:1:N, 'xticklabel',N:-1:1 , 'ylim',[0 4],'ytick',0:1:4 )
    xlabel('signal duration')
    ylabel('d-prime')
end

end