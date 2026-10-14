function out = st_ref_arima_realgdp(root)
% ARMA(1,1) with constant on macrodata realgdp, exact Gaussian ML via Econometrics Toolbox estimate.
T = readtable(fullfile(root, 'data', 'examples', 'timeseries', 'macrodata.csv'));
y = T.realgdp;
[E, ~, logL] = estimate(arima(1, 0, 1), y, 'Display', 'off');
out.y = y;
out.constant = E.Constant;
out.ar = E.AR{1};
out.ma = E.MA{1};
out.variance = E.Variance;
out.mean = E.Constant / (1 - E.AR{1});
out.loglik = logL;
end
