args root outdir
adopath + "`root'/tests/external_benchmarks/stata/drivers"
import delimited "`root'/data/examples/regression/longley.csv", clear asdouble case(preserve) varnames(1)
pca GNPDEFL GNP UNEMP ARMED POP YEAR, components(6)
matrix ev = e(Ev)
matrix L = e(L)
_st_export ev "`outdir'/eigenvalues.csv"
_st_export L "`outdir'/loadings.csv"
exit, clear
