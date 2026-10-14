args root outdir input_csv
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`input_csv'", clear asdouble varnames(1)
gen t = _n
tsset t
matrix R = J(18, 4, .)
forvalues h = 0/17 {
    quietly regress F`h'.lcpi rr_shock dlrgdp dlcpi dstir, vce(hc3)
    matrix R[`h'+1, 1] = `h'
    matrix R[`h'+1, 2] = _b[rr_shock]
    matrix R[`h'+1, 3] = _se[rr_shock]
    matrix R[`h'+1, 4] = e(N)
}
_st_export R "`outdir'/irf.csv"
exit, clear
