# R DataFrame
- Matrix of values
- If the column name is `colname`, then access that row as a regular [[R vector]] using `d$colname`
- Access a single element using the form `d[row, col]`
- Lots of different ways to express the rows: e.g. `d[d$age >= 18, ]` will select all rows where the age is >= 18 and all corresponding columns