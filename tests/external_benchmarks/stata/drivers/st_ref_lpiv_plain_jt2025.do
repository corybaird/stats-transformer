args root outdir input_csv
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`input_csv'", clear asdouble varnames(1)
gen t = _n
tsset t
matrix R = J(49, 5, .)
forvalues h = 0/48 {
    quietly ivregress 2sls F`h'.urate infl (ffr = rrcgshock), small
    matrix R[`h'+1, 1] = `h'
    matrix R[`h'+1, 2] = _b[ffr]
    matrix R[`h'+1, 3] = _se[ffr]
    matrix R[`h'+1, 4] = e(N)
    quietly ivregress 2sls F`h'.urate infl (ffr = rrcgshock)
    matrix R[`h'+1, 5] = _se[ffr]
}
_st_export R "`outdir'/irf.csv"
exit, clear
