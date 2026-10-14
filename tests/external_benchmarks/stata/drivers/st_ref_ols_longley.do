args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/regression/longley.csv", clear asdouble
gen double one = 1
regress totemp gnpdefl gnp unemp armed pop year
matrix b = e(b)'
matrix V = e(V)
matrix se_classical = vecdiag(V)'
forvalues i = 1/`=rowsof(se_classical)' {
    matrix se_classical[`i',1] = sqrt(se_classical[`i',1])
}
predict double u, resid
mkmat u, matrix(resid)
_st_export b "`outdir'/beta.csv"
_st_export se_classical "`outdir'/se_classical.csv"
_st_export resid "`outdir'/resid.csv"
foreach t in hc2 hc3 {
    quietly regress totemp gnpdefl gnp unemp armed pop year, vce(`t')
    matrix s = vecdiag(e(V))'
    forvalues i = 1/`=rowsof(s)' {
        matrix s[`i',1] = sqrt(s[`i',1])
    }
    _st_export s "`outdir'/se_`t'.csv"
}
quietly regress totemp gnpdefl gnp unemp armed pop year, vce(robust)
matrix s = vecdiag(e(V))'
forvalues i = 1/`=rowsof(s)' {
    matrix s[`i',1] = sqrt(s[`i',1])
}
_st_export s "`outdir'/se_hc1.csv"
exit, clear
