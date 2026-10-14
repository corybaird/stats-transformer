function out = st_ref_lpiv_jt2025(root)
% Mirrors Replic/JT2025/GO_JT2025.m Example 6 (LP-IV, FFR instrumented by RRCGShock and 6 lags, 1985m1 to 2000m1).
tb = fullfile(root, 'references', 'matlab_benchmarks');
p = strjoin(cellfun(@(d) genpath(fullfile(tb, d)), {'VAR', 'Utils', 'Stats', 'Auxiliary'}, 'UniformOutput', false), pathsep);
addpath(p);
cleanup = onCleanup(@() rmpath(p));
f = fullfile(root, 'data', 'examples', 'matlab_examples', 'JT2025_Data.xlsx');
xt = readcell(f, 'Sheet', 'Ex6');
data = readmatrix(f, 'Sheet', 'Ex6', 'Range', 'B3');
dates = xt(3:end, 1);
mnem = xt(2, 2:end);
n = size(data, 1);
yr = zeros(n, 1);
mo = zeros(n, 1);
for ii = 1:n
    k = strfind(dates{ii}, 'm');
    yr(ii) = str2double(dates{ii}(1:k-1));
    mo(ii) = str2double(dates{ii}(k+1:end));
end
i0 = find(yr == 1985 & mo == 1);
i1 = find(yr == 2000 & mo == 1);
col = @(m) data(i0:i1, strcmp(mnem, m));
urate = col('urate');
infl = col('infl');
ffr = col('ffr');
z = col('RRCGShock');
opt = LPoption;
opt.nsteps = 49;
opt.longdiff = 1;
opt.impact = 1;
opt.pctg = 95;
opt.IV = z;
opt.nlag_iv = 6;
LP = LPmodel(urate, ffr, [urate infl ffr], 6, 1, opt);
out.ENDO = urate;
out.TREAT = ffr;
out.CTRL = [urate infl ffr];
out.IV = z;
out.rows_used = [i0 i1];
out.IR = LP.IR;
out.INF = LP.INF;
out.SUP = LP.SUP;
out.jtest_df = LP.jtest_df;
out.jtest_pval = LP.jtest_pval;
end
