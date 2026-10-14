args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/timeseries/macrodata.csv", clear asdouble case(preserve) varnames(1)
gen t = _n
tsset t
arima realgdp, arima(1,0,1)
matrix b = e(b)'
_st_export b "`outdir'/b.csv"
matrix se = vecdiag(e(V))'
forvalues i = 1/`=rowsof(se)' {
    matrix se[`i',1] = sqrt(se[`i',1])
}
_st_export se "`outdir'/se.csv"
matrix n = (e(N), e(ll))
_st_export n "`outdir'/scalars.csv"
exit, clear
