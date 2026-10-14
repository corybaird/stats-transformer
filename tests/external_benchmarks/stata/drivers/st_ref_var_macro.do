args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/timeseries/macrodata.csv", clear asdouble case(preserve) varnames(1)
gen t = _n
tsset t
var realgdp realcons realinv, lags(1/2)
matrix b = e(b)
_st_export b "`outdir'/b_rowvec.csv"
matrix Sigma = e(Sigma)
_st_export Sigma "`outdir'/Sigma_ml.csv"
matrix n = (e(N), e(T), e(df_m), e(ll), e(aic), e(sbic), e(hqic), e(detsig))
_st_export n "`outdir'/scalars.csv"
forvalues j = 1/3 {
    matrix bj`j' = e(b)[1, `=(`j'-1)*7+1'..`=`j'*7']'
    _st_export bj`j' "`outdir'/eq`j'_coef.csv"
}
exit, clear
