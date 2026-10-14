function out = st_ref_gk2015_short(root)
% Mirrors Replic/GK2015/GO_GK2015.m sections 2, 4 and 5 with inference = 0 and no plotting.
tb = fullfile(root, 'references', 'matlab_benchmarks');
p = strjoin(cellfun(@(d) genpath(fullfile(tb, d)), {'VAR', 'Utils', 'Stats', 'Auxiliary'}, 'UniformOutput', false), pathsep);
addpath(p);
cleanup = onCleanup(@() rmpath(p));
raw = readcell(fullfile(root, 'data', 'examples', 'matlab_examples', 'GK2015_Data.xlsx'), 'Sheet', 'Sheet1');
mnem = raw(2, 2:end);
data = Num2NaN(cellfun(@double, raw(3:end, 2:end)));
VARmnem = {'gs1'; 'logcpi'; 'logip'; 'ebp'};
nobs = size(data, 1);
ENDO = nan(nobs, numel(VARmnem));
for ii = 1:numel(VARmnem)
    ENDO(:, ii) = data(:, strcmp(mnem, VARmnem{ii}));
end
nlags = 12;
detc = 1;
VARopt = VARoption;
VARopt.mnem = VARmnem;
VARopt.ident = 'short';
VARopt.nsteps = 48;
VARopt.inference = 0;
VAR = VARmodel(ENDO, nlags, detc, VARopt);
out.ENDO = ENDO;
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
