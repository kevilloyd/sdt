function plotter(bel, ax)
%clf;
if nargin<2
    figure;
    ax = gca;
end
[T,~] = size(bel);
bel_tot = bel(1:T-1,1:3)+bel(1:T-1,4:6);
plot(ax, 1:T-1, bel_tot, 'linewidth',2);
hold on
leg.FontSize = 10;
att = sum(bel(:,4:6),2);
plot(ax,1:T-1, att(2:end),'k--','linewidth',2);
legend('pre','on','post','att','Location','West');


