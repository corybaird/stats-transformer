args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/timeseries/macrodata.csv", clear asdouble case(preserve) varnames(1)
gen t = _n
tsset t
vec realgdp realcons realinv, rank(1) lags(2) trend(constant)
matrix alpha = e(alpha)
matrix beta = e(beta)
matrix Pi = e(pi)
matrix Sigma = e(omega)
_st_export alpha "`outdir'/alpha.csv"
_st_export beta "`outdir'/beta.csv"
_st_export Pi "`outdir'/Pi.csv"
_st_export Sigma "`outdir'/Sigma.csv"
matrix Va = e(V_alpha)
_st_export Va "`outdir'/V_alpha.csv"
matrix Vb = e(V_beta)
_st_export Vb "`outdir'/V_beta.csv"
matrix bfull = e(b)
_st_export bfull "`outdir'/b_short_run.csv"
matrix n = (e(N), e(ll), e(aic), e(sbic), e(hqic), e(detsig_ml))
_st_export n "`outdir'/scalars.csv"
exit, clear
