args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/tests/integration/var/canada.csv", clear asdouble case(preserve) varnames(1)
gen t = _n
tsset t
matrix Ap = (.,0,0,0 \ .,.,0,0 \ 0,0,.,0 \ .,0,0,.)
matrix Bp = I(4)
svar e prod rw U, lags(1/2) aeq(Ap) beq(Bp) iterate(1000) nrtolerance(1e-12)
matrix A = e(A)
matrix S = e(Sigma)
matrix imp = inv(A)
_st_export A "`outdir'/A.csv"
_st_export imp "`outdir'/impact.csv"
_st_export S "`outdir'/Sigma_ml.csv"
matrix n = (e(N), e(ll), e(chi2_oid), e(df_oid), e(converged))
_st_export n "`outdir'/scalars.csv"
exit, clear
