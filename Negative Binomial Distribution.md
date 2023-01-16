# Negative Binomial Distribution
- Models the number of successes before a specified number of failures
- Given a number of failures $r$, $k$ the number of successes, and $p$ the probability of failure, the PMF is given by
$$
p(k;r,p) = \binom{k + r - 1}{r-1}(1-p)^k p^r
$$