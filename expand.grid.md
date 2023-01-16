# `expand.grid`
Returns an [[R DataFrame]] whose entries contain all possible combinations of the input rows:
```R
expand.grid(1:3, 2:5)

A data.frame: 12 × 2
Var1   Var2
<int> <int>
1       2
2       2
3       2
1       3
2       3
3       3
1       4
2       4
3       4
1       5
2       5
3       5
```

You can also name the columns:
```R
expand.grid(x=c(1, 2, 3), y=c(5, 6))
A data.frame: 6 × 2
x       y
<dbl> <dbl>
1       4
2       4
3       4
1       5
2       5
3       5
```