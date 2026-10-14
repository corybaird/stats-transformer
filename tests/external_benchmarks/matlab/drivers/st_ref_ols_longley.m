function out = st_ref_ols_longley(root)
% Longley OLS reference: beta = X\y, classical and HC3 standard errors computed by hand.
% X'X is numerically singular here (cond about 1e20), so (X'X)^-1 is built from the QR factor R as inv(R)*inv(R)'.
T = readtable(fullfile(root, 'data', 'examples', 'regression', 'longley.csv'));
y = T.TOTEMP;
X = [ones(height(T), 1), T.GNPDEFL, T.GNP, T.UNEMP, T.ARMED, T.POP, T.YEAR];
[n, k] = size(X);
beta = X \ y;
resid = y - X * beta;
s2 = (resid' * resid) / (n - k);
[~, R] = qr(X, 0);
Ri = inv(R);
XtXi = Ri * Ri';
se_classical = sqrt(diag(s2 * XtXi));
h = sum((X * Ri) .^ 2, 2);
w = (resid ./ (1 - h)) .^ 2;
V_hc3 = XtXi * (X' * (X .* w)) * XtXi;
se_hc3 = sqrt(diag(V_hc3));
out.X = X;
out.y = y;
out.beta = beta;
out.resid = resid;
out.se_classical = se_classical;
out.se_hc3 = se_hc3;
end
