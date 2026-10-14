args root outdir input_csv
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`input_csv'", clear asdouble varnames(1)
gen t = _n
tsset t
matrix C = (., 0 \ ., .)
svar y u, lags(1/8) lreq(C)
matrix B = e(B)
matrix A = e(A)
matrix Cl = e(lreq)
matrix bvar = e(b_var)
matrix Sigma = e(Sigma)
_st_export B "`outdir'/B_impact.csv"
_st_export A "`outdir'/A.csv"
_st_export Cl "`outdir'/lreq.csv"
_st_export bvar "`outdir'/b_var.csv"
_st_export Sigma "`outdir'/Sigma_ml.csv"
exit, clear
