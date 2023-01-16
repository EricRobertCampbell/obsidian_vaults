# Hypergeometric Distribution
- Probability of $k$  successes in $n$  draws, *without* replacement, from a population of size $N$  of which $K$  would be considered 'successes'
- E.g. you have an urn with 100 balls, 25 black and 75 white. Then the parameters for drawing 5 black in 10 draws:
	- $k = 5$
	- $n=10$
	- $N=100$
	- $K=25$

- PMF is 
$$
p_X(k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}
$$
- Almost the same as the [[Binomial Distribution]], except that that one is draws *with* replacement