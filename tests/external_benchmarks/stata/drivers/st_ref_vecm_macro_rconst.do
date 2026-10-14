args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/timeseries/macrodata.csv", clear asdouble case(preserve) varnames(1)
gen t = _n
tsset t
vec realgdp realcons realinv, rank(1) lags(2) trend(rconstant)
matrix alpha = e(alpha)
matrix beta = e(beta)
matrix Sigma = e(omega)
matrix bfull = e(b)
_st_export alpha "`outdir'/alpha.csv"
_st_export beta "`outdir'/beta.csv"
_st_export Sigma "`outdir'/Sigma.csv"
_st_export bfull "`outdir'/b_short_run.csv"
matrix n = (e(N), e(ll))
_st_export n "`outdir'/scalars.csv"
exit, clear
