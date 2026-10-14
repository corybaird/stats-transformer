function out = st_ref_vecm_macro(root)
% Johansen VECM, rank 1, one lagged difference, Econometrics Toolbox estimate with Model H1 (unrestricted constant) and H1* (constant in the cointegrating relation).
T = readtable(fullfile(root, 'data', 'examples', 'timeseries', 'macrodata.csv'));
Y = [T.realgdp T.realcons T.realinv];
Mdl = vecm(3, 1, 1);
E1 = estimate(Mdl, Y, 'Model', 'H1');
E2 = estimate(Mdl, Y, 'Model', 'H1*');
out.ENDO = Y;
out.h1_alpha = E1.Adjustment;
out.h1_beta = E1.Cointegration;
out.h1_constant = E1.Constant;
out.h1_short_run = E1.ShortRun{1};
out.h1_covariance = E1.Covariance;
out.h1star_alpha = E2.Adjustment;
out.h1star_beta = E2.Cointegration;
out.h1star_beta_constant = E2.CointegrationConstant;
out.h1star_constant = E2.Constant;
out.h1star_short_run = E2.ShortRun{1};
out.h1star_covariance = E2.Covariance;
end
