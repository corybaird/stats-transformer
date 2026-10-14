args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/regression/grunfeld.csv", clear asdouble case(preserve) varnames(1)
encode firm, gen(firm_id)
xtset firm_id year
xtreg invest value capital, fe
matrix b = e(b)'
matrix V = e(V)
_st_export b "`outdir'/beta_fe.csv"
matrix se = vecdiag(V)'
forvalues i = 1/`=rowsof(se)' {
    matrix se[`i',1] = sqrt(se[`i',1])
}
_st_export se "`outdir'/se_fe.csv"
matrix n = (e(N), e(N_g), e(r2_w), e(sigma_u), e(sigma_e))
_st_export n "`outdir'/fe_scalars.csv"
quietly xtreg invest value capital, fe vce(cluster firm_id)
matrix s = vecdiag(e(V))'
forvalues i = 1/`=rowsof(s)' {
    matrix s[`i',1] = sqrt(s[`i',1])
}
_st_export s "`outdir'/se_fe_cluster.csv"
exit, clear
