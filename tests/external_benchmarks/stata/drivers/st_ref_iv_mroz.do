args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/regression/mroz.csv", clear asdouble case(preserve) varnames(1)
destring lwage, replace force
keep if inlf == 1
ivregress 2sls lwage exper expersq (educ = motheduc fatheduc)
matrix b = e(b)'
matrix V = e(V)
_st_export b "`outdir'/beta.csv"
matrix se = vecdiag(V)'
forvalues i = 1/`=rowsof(se)' {
    matrix se[`i',1] = sqrt(se[`i',1])
}
_st_export se "`outdir'/se.csv"
matrix n = (e(N))
_st_export n "`outdir'/nobs.csv"
quietly ivregress 2sls lwage exper expersq (educ = motheduc fatheduc), vce(robust)
matrix s = vecdiag(e(V))'
forvalues i = 1/`=rowsof(s)' {
    matrix s[`i',1] = sqrt(s[`i',1])
}
_st_export s "`outdir'/se_robust.csv"
exit, clear
