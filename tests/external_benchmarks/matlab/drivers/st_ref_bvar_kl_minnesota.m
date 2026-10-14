function out = st_ref_bvar_kl_minnesota(root)
% Mirrors Kilian and Luetkepohl (2017) Code_Kilian/5/figure5_2: Minnesota prior with Sigma fixed at diag(sigma_sq).
% With Sigma fixed the posterior of each equation is Gaussian, so the analytic posterior mean replaces the Gibbs average.
kl = fullfile(root, 'tests', 'external_benchmarks', 'matlab', 'third_party', 'kilian_lutkepohl_2017', '5', 'figure5_2');
addpath(kl);
cleanup = onCleanup(@() rmpath(kl));
Yraw = load(fullfile(kl, 'Yraw.dat'));
constant = 1;
p = 4;
[Traw, M] = size(Yraw);
Ylag = mlag2(Yraw, p);
X = [ones(Traw-p, 1) Ylag(p+1:Traw, :)];
K = size(X, 2);
Y = Yraw(p+1:Traw, :);
A_OLS = inv(X' * X) * (X' * Y);
prior_hyper_Minnesota;
a_post = zeros(K * M, 1);
V_post_diag = zeros(K * M, 1);
for i = 1:M
    idx = (i-1)*K+1:i*K;
    Vp = V_prior(idx, idx);
    V_post = inv(inv(Vp) + inv(SIGMA(i,i)) * (X' * X));
    a_post(idx) = V_post * (inv(Vp) * a_prior(idx) + inv(SIGMA(i,i)) * X' * Y(:, i));
    V_post_diag(idx) = diag(V_post);
end
out.ENDO = Yraw;
out.set_nlags = p;
out.set_const = constant;
out.A_OLS = A_OLS;
out.sigma_sq = sigma_sq;
out.a_prior = a_prior;
out.V_prior_diag = diag(V_prior);
out.A_post_mean = reshape(a_post, K, M);
out.V_post_diag = reshape(V_post_diag, K, M);
out.lambda = a_bar_1;
out.theta = a_bar_2;
out.intercept_var = a_bar_3;
end
