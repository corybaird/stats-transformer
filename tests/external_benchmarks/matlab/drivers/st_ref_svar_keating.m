function out = st_ref_svar_keating(root)
% Mirrors Kilian and Luetkepohl (2017) Code_Kilian/9/ml_nonrecursive (Keating 1992 model, p = 4).
% data.m is reproduced with absolute paths; fsolve tolerances are tightened from 1e-4 to 1e-12.
kl = fullfile(root, 'tests', 'external_benchmarks', 'matlab', 'third_party', 'kilian_lutkepohl_2017', '9', 'ml_nonrecursive');
addpath(kl);
cleanup = onCleanup(@() rmpath(kl));
gnpdeflator = load(fullfile(kl, 'gnpdeflator.txt'));
infl = dif(log(gnpdeflator(:, 3))) * 100;
realgnp = load(fullfile(kl, 'realgnp.txt'));
drgdp = dif(log(realgnp(:, 3))) * 100;
fedfunds = load(fullfile(kl, 'fedfunds.txt'));
irate = [];
for i = 1:3:length(fedfunds(:, 3))
    irate = [irate; mean(fedfunds(i:i+2, 3))];
end
m1 = load(fullfile(kl, 'm1.txt'));
m1 = log(m1(3:end, 3));
dmoney = [];
for i = 4:3:length(m1)
    dmoney = [dmoney; (m1(i, 1) - m1(i-3, 1)) * 100];
end
y = [infl(19:end, 1) drgdp(19:end, 1) irate(19:end, 1) dmoney];
[t, q] = size(y);
p = 4;
[A, ~, U, V] = olsvarc(y, p);
SIGMAu = U * U' / (t - p);
SIGMAu = SIGMAu(1:q, 1:q);
options = optimset('TolX', 1e-12, 'TolFun', 1e-12, 'MaxFunEvals', 1e+6, 'MaxIter', 10000, 'Display', 'off');
start = [eye(q) 1.5 * ones(q, 1)];
eq_fun = @(g) [reshape(inv(g(1:q,1:q)) * diag(g(:,q+1)) * inv(g(1:q,1:q))' - SIGMAu, [], 1); g(1,2); g(1,3); g(1,4); g(3,1); g(3,2); g(4,1) - g(4,2); g(1,1) - 1; g(2,2) - 1; g(3,3) - 1; g(4,4) - 1];
zero_fun = @(g) [reshape(inv(g(1:q,1:q)) * diag(g(:,q+1)) * inv(g(1:q,1:q))' - SIGMAu, [], 1); g(1,2); g(1,3); g(1,4); g(3,1); g(3,2); g(4,1); g(1,1) - 1; g(2,2) - 1; g(3,3) - 1; g(4,4) - 1];
[g_eq, ~, flag_eq] = fsolve(eq_fun, start, options);
[g_zero, ~, flag_zero] = fsolve(zero_fun, start, options);
out.ENDO = y;
out.set_nlags = p;
out.A_lags = A(1:q, :);
out.const = V(1:q);
out.resid = U(1:q, :)';
out.SIGMAu = SIGMAu;
out.B0_equality = g_eq(1:q, 1:q);
out.SIGMAw_equality = g_eq(:, q+1);
out.impact_equality = inv(g_eq(1:q, 1:q)) * diag(sqrt(g_eq(:, q+1)));
out.exitflag_equality = flag_eq;
out.B0_zero = g_zero(1:q, 1:q);
out.SIGMAw_zero = g_zero(:, q+1);
out.impact_zero = inv(g_zero(1:q, 1:q)) * diag(sqrt(g_zero(:, q+1)));
out.exitflag_zero = flag_zero;
end
