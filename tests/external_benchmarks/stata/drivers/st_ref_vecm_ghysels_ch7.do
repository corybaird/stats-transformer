args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
local base "`root'/data/examples/timeseries/ghysels_ch7"
import delimited "`base'/simulated_cointegration.csv", clear asdouble varnames(1)
keep in 1/400
gen t = _n
tsset t
vec x y, rank(1) lags(2) trend(rconstant)
matrix a = e(alpha)
matrix b = e(beta)
matrix S = e(omega)
_st_export a "`outdir'/sim_alpha.csv"
_st_export b "`outdir'/sim_beta.csv"
_st_export S "`outdir'/sim_Sigma.csv"
import delimited "`base'/dataonly_uktermstructure.csv", clear asdouble varnames(1)
keep in 1/216
gen t = _n
tsset t
vec r1 r3 r6 r24, rank(1) lags(3) trend(rconstant)
matrix a = e(alpha)
matrix b = e(beta)
matrix S = e(omega)
_st_export a "`outdir'/uk_r1_alpha.csv"
_st_export b "`outdir'/uk_r1_beta.csv"
_st_export S "`outdir'/uk_r1_Sigma.csv"
vec r1 r3 r6 r24, rank(3) lags(3) trend(rconstant)
matrix a = e(alpha)
matrix b = e(beta)
_st_export a "`outdir'/uk_r3_alpha.csv"
_st_export b "`outdir'/uk_r3_beta.csv"
import delimited "`base'/dataonly_uslei.csv", clear asdouble varnames(1)
keep in 313/540
gen t = _n
tsset t
vec cci cli, rank(1) lags(2) trend(rconstant)
matrix a = e(alpha)
matrix b = e(beta)
matrix S = e(omega)
_st_export a "`outdir'/lei_alpha.csv"
_st_export b "`outdir'/lei_beta.csv"
_st_export S "`outdir'/lei_Sigma.csv"
exit, clear
