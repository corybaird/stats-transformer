function out = st_ref_lp_jt2025(root)
% Mirrors Replic/JT2025/GO_JT2025.m Example 5 (LP-OLS, Romer-Romer shock, 1985q1 to 2007q4).
tb = fullfile(root, 'references', 'matlab_benchmarks');
p = strjoin(cellfun(@(d) genpath(fullfile(tb, d)), {'VAR', 'Utils', 'Stats', 'Auxiliary'}, 'UniformOutput', false), pathsep);
addpath(p);
cleanup = onCleanup(@() rmpath(p));
f = fullfile(root, 'data', 'examples', 'matlab_examples', 'JT2025_Data.xlsx');
xt = readcell(f, 'Sheet', 'Ex5');
data = readmatrix(f, 'Sheet', 'Ex5', 'Range', 'B3');
dates = xt(3:end, 1);
mnem = xt(2, 2:end);
n = size(data, 1);
yr = zeros(n, 1);
qt = zeros(n, 1);
for ii = 1:n
    k = strfind(dates{ii}, 'q');
    yr(ii) = str2double(dates{ii}(1:k-1));
    qt(ii) = str2double(dates{ii}(k+1:end));
end
i0 = find(yr == 1985 & qt == 1);
i1 = find(yr == 2007 & qt == 4);
col = @(m) data(i0:i1, strcmp(mnem, m));
lcpi = 100 * col('lcpi');
rr_shock = col('rr_shock');
CTRL = [col('dlrgdp') col('dlcpi') col('dstir')];
opt = LPoption;
opt.nsteps = 18;
opt.longdiff = 1;
opt.impact = 1;
opt.pctg = 95;
opt.IV = [];
LP = LPmodel(lcpi, rr_shock, CTRL, 4, 1, opt);
out.ENDO = lcpi;
out.TREAT = rr_shock;
out.CTRL = CTRL;
out.rows_used = [i0 i1];
out.IR = LP.IR;
out.INF = LP.INF;
out.SUP = LP.SUP;
end
