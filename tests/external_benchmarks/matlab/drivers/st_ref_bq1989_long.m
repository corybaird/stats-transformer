function out = st_ref_bq1989_long(root)
% Mirrors Replic/BQ1989/GO_BQ1989.m sections 2 and 4 to 5 with inference = 0 and no plotting.
tb = fullfile(root, 'references', 'matlab_benchmarks');
p = strjoin(cellfun(@(d) genpath(fullfile(tb, d)), {'VAR', 'Utils', 'Stats', 'Auxiliary'}, 'UniformOutput', false), pathsep);
addpath(p);
cleanup = onCleanup(@() rmpath(p));
raw = readcell(fullfile(root, 'data', 'examples', 'matlab_examples', 'BQ1989_Data.xlsx'), 'Sheet', 'Sheet1');
mnem = raw(2, 2:end);
X = cellfun(@double, raw(3:end, 2:end));
detc = 1;
nlags = 8;
VARopt = VARoption;
VARopt.mnem = mnem;
VARopt.nsteps = 40;
VARopt.ident = 'long';
VARopt.inference = 0;
VAR = VARmodel(X, nlags, detc, VARopt);
out.ENDO = X;
out.set_nlags = nlags;
out.set_const = detc;
out.set_nsteps = VARopt.nsteps;
out.set_inference = VARopt.inference;
out.Ft = VAR.Ft;
out.Sigma = VAR.sigma;
out.resid = VAR.resid;
out.B = VAR.B;
out.IR = VAR.IR;
end
