program define _st_export
    args mname path
    tempname fh
    local r = rowsof(`mname')
    local c = colsof(`mname')
    file open `fh' using "`path'", write text replace
    forvalues i = 1/`r' {
        local line ""
        forvalues j = 1/`c' {
            local v = `mname'[`i',`j']
            local s : display %24.17g `v'
            local s = trim("`s'")
            if `j' > 1 local line "`line',"
            local line "`line'`s'"
        }
        file write `fh' "`line'" _n
    }
    file close `fh'
    local rn : rowfullnames `mname'
    local cn : colfullnames `mname'
    file open `fh' using "`path'.labels", write text replace
    file write `fh' "`rn'" _n
    file write `fh' "`cn'" _n
    file close `fh'
end
