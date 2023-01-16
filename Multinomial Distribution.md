# Multinomial Distribution
- Generalization of the[[Binomial Distribution]] to more than two outcomes
- Say you have a dice with $n$ faces, each one appearing with probability $p_i$. Then the probability of seeing $x_1$  face 1s, $x_2$  face 2s, ... is 
$$
p(x_1, x_2, \ldots, x_2) = \begin{cases}
\frac{n!}{x_1! x_2! \dots x_n!} p_1 ^{x_1} \cdot p_2 ^ {x_2} \cdots p_n ^ {x_n} & \sum x_i = n \\
0 & \text{otherwise}
\end{cases}
$$
