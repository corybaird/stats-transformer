args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/discrete/spector.csv", clear asdouble case(preserve) varnames(1)
logit GRADE GPA TUCE PSI
matrix b = e(b)'
matrix V = e(V)
_st_export b "`outdir'/beta.csv"
matrix se = vecdiag(V)'
forvalues i = 1/`=rowsof(se)' {
    matrix se[`i',1] = sqrt(se[`i',1])
}
_st_export se "`outdir'/se.csv"
matrix ll = (e(ll), e(ll_0), e(N))
_st_export ll "`outdir'/loglik.csv"
exit, clear
