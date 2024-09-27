# Chapter 12 - Monsters and Mixtures

This chapter is about making likelihood / link functions composed of different pieces of the ones we've looked at in previous chapters.

Three common and useful models:
1. Models for handling over-dispersion - extensions to the binomial and Poisson models to cope with unmeasured sources of variations
1. **Zero-inflated** and **zero-augmented** models, which mixes a binary event with an ordinary GLM likelihood like a Poisson or binomial
1. Ordered categorical models, useful for categorical outcomes with a fixed ordering. This model is built by merging a categorical likelihood function with a special kind of link function, usually a **cumulative link**

## 1 Over-dispersed counts

In [[SR - Chapter 7 - Ulysses' Compass]], we saw that models based on the [[Normal Distribution]] can be overly sensitive to outliers.

The problem isn't the outliers - instead, it's the fact that the counts arise from a mixture of different processes, which can result in thicker tails than we might naively expect.

When counts are more variable than a pure process would suggest, we say that they exhibit **over-dispersion**. The variance of a variable is sometimes called its **dispersion**.

For instance, in a [[Binomial Distribution|binomial]] variable the expected variance is $Np / (1 - p)$. If we observe a greater variance than that, then it might be that some unobserved variable is producing additional dispersion.

Our model might be good enough for predictive work, even with the over-dispersion. However, we might be missing out on an important effect. It's worth the time to dig into the cause of the over-dispersion.

In this chapter we'll deal with continuous mixture models in which a linear model is attached not to the observations, but to the distribution of the observations.

In [[SR - Chapter 13 - Models with Memory]], we'll see how to use [[Multilevel models]] that estimte both the residuals of each observation and the distribution of those residuals. It's often easier to use multilevel models than mixture models because they can handle additional kinds of heterogeneity.

### 1.1 Beta-binomial

A **Beta-binomial** model is a mixture of binomial distributions. Assumption is that each binomial count has some probability of success, and then we estimate the distribution of probabilities of success instead of a single probability of success. Any predictor variables describe the shape of the distribution.

For example, let's look at the `UCBAdmit` data. The model was over-dispersed when we ignored the department since there was a lot of variability coming from there. Let's see how a beta-binomial model can pick up on this, even when we ignore department as a predictor.

The beta-binomial model will assume that each observed count on each row of the data table has its own unique, unobserved probability of admissions. These probabilities of admission have their own common distribution. We describe the distribution of these probabilities using a **beta distribution**. One advantage of using the **beta distribution** is that we can derive a closed-form solution for the likelihood function.

A **beta distribution** has two parameters: and average probability $\bar{p}$ and a shape parameter $\theta$. The shape parameter $\theta$ describes how spread out the distribution is. When $\theta = 2$, all probabilities are equally likely. When $\theta > 2$, the range of probabilities gets limited to a spike around the mean. When $\theta < 2$, the values are so dispersed that they actually moave *awa* from the mean and make extreme values like 0 or 1 more likely.

We're going to make a model where the predictor variables change $\bar{p}$, the mean.

$$
\begin{align*}
    A_i &\sim \text{BetaBinomial}(N_i, \bar{p_i}, \theta) \\
    \text{logit}(\bar{p_i}) &= \alpha_{\text{GID}[i]} \\
    \alpha_j &\sim \text{Normal}(0, 1.5) \\
    \theta &= \phi + 2 \\
    \phi &\sim \text{Exponential}(1)
\end{align*}
$$

$A$ is the admittance, $N$ is the number of applications, and $\text{GID}[i]$ is the gender index - 1 for male, 2 for female.

Here we have a trick in the prior on $\theta$. We want the value to be greater than 2, so we add two to the exponential distribution (lowest value: 0) to make our own distribution have a minimum value of 2.


```R
library(rethinking)
library(ggplot2)
library(gtools)
library(repr)

options(repr.plot.width = 16, repr.plot.height = 8)
```

```R
data(UCBadmit)

d <- UCBadmit
d$gid <- ifelse(d$applicant.gender == 'male', 1L, 2L)
dat <- list(A = d$admit, N = d$applications, gid = d$gid)
m12.1 <- ulam(
    alist(
        A ~ dbetabinom(N, pbar, theta),
        logit(pbar) <- a[gid],
        a[gid] ~ dnorm(0, 1.5),
        transpars> theta <<- phi + 2,
        phi ~ dexp(1)
    ),
    data = dat,
    chains = 4
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.6 seconds.

```R
# looking at the contrast first
post <- extract.samples(m12.1)
post$da <- post$a[,1] - post$a[,2]
precis(post, depth=2)
```

<table class="dataframe">
<caption>A precis: 5 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td>-0.4421293</td><td>0.4063700</td><td>-1.08906100</td><td>0.2157494</td><td>▁▁▇▇▂▁▁   </td></tr>
	<tr><th scope=row>a[2]</th><td>-0.3262416</td><td>0.3895973</td><td>-0.93744867</td><td>0.2996487</td><td>▁▁▅▇▂▁▁   </td></tr>
	<tr><th scope=row>phi</th><td> 1.0485135</td><td>0.8184643</td><td> 0.09524084</td><td>2.5758021</td><td>▇▇▅▃▂▁▁▁▁▁</td></tr>
	<tr><th scope=row>theta</th><td> 3.0485135</td><td>0.8184643</td><td> 2.09524180</td><td>4.5758021</td><td>▇▇▅▃▂▁▁▁▁▁</td></tr>
	<tr><th scope=row>da</th><td>-0.1158877</td><td>0.5732637</td><td>-1.00963352</td><td>0.8303310</td><td>▁▁▁▅▇▇▂▁▁ </td></tr>
</tbody>
</table>



The parameter $a[1]$ is the log-odds of admission for male applications. It is slightly lower than for female ones, but from the different $da$ we can see that it is highly uncertain.

Recall that in **Chapter 11**, a binomial model of the data that omitted the department resulted in a spurious indication that the female admission rate was lower than the male one. So why doesn't this model have the same problem?

The beta-binomial model allows each row in the data (each combination of department + gender) to have its own unobserved intercept. These unobserved intercepts are sampled from a beta distribution with mean $\bar{p_i}$ and dispersion $\theta$. We can plot this distribution.


```R
# female curves
gid <- 2
ps <- seq(0, 1, length.out = 100)

mean_curve <- data.frame(
    x = ps,
    y = dbeta2(ps, mean(logistic(post$a[,gid])), mean(post$theta))
)

graph <- ggplot() +
    geom_line(data = mean_curve, mapping = aes(x, y), linewidth = 2)

# now add some distributions sampled from the posterior
for (i in 1:50) {
    p <- logistic(post$a[i, gid])
    theta <- post$theta[i]
    sample_curve_data <- data.frame(
        x = ps,
        y = dbeta2(ps, p, theta)
    )
    graph <- graph +
        geom_line(data = sample_curve_data, mapping = aes(x, y), alpha = 0.2)
}

print(graph)
```


    
![png](sr_chapter_12_images/output_5_0.png)
    


What this shows is that although the posterior is slanted toward low rates of admission, there is a lot of variation - it is plausible that there will be departments that wil admit almost no or almost all of the applicants! The model is no longer 'tricked' by department variation into a false inference about gender.

To get a sense of how the beta distribution of probabilities of admission influences predicted counts of applications admitted, let's look at the posterior check:


```R
par(bg = 'white')
postcheck(m12.1)
```


    
![png](sr_chapter_12_images/output_7_0.png)
    


The blue circles are the empirical rates of admission. The hollow circles and lines are the posterior mean $\bar{p}$ and the 89% credible interval, and the + symbols are the 89% intervale for admission numbers.

There is a lot of dispersion here. The model doesn't know that it comes from the department (that data isn't in the model), but it can see that there is a lot of dispersion and account for it.

## 2 Negative-binomial or gamma-Poisson

The negative-binomial or gamma-Poisson model assumes that each Poisson count observation has its own rate. It estimates the shape of a gamma distribution to describe the Poisson rates across the different cases. Predictor variables adjust the shape of the distribution, not the expected value of each observation.

The gamma-Poisson is very much like the beta-binomial, except that here the beta distribution of rates is replaced by the gamma distribution of rates / expected values. We use the gamma because the math works out nicely; there's an analytical formula.

The gamma-Poisson distribution has two parameters, one for the mean (rate) and another for the dispersion (scale) of the rates across cases.

$$
y_i \sim \text{Gamma-Poisson}(\lambda_i, \phi)
$$

The $\lambda$ parameter behaves like the rate of an ordinary Poisson. The $\phi$ parameter must be positive and cotrols the variance. The variance of a gamma-Poisson is $\lambda + \lambda^2 / \phi$ - the larger $\phi$, the more similar the results are to a pure Poisson process.

Let's try it out with the Oceanic Tools example. In that dataset, Hawaii was very influential. However, it will be far less influential in the equivalent gamma-Poisson model, because that model expects more variation; thus, the 'outlier' point has less of an effect on the overall trend.


```R
data(Kline)
d <- Kline
d$P <- standardize(log(d$population))
d$contact_id <- ifelse(d$contact == 'high', 2L, 1L)

data2 <- list(
    T = d$total_tools,
    P = d$population,
    cid = d$contact_id
)

m12.2 <- ulam(
    alist(
        T ~ dgampois(lambda, phi),
        lambda <- exp(a[cid]) * P ^ b[cid] / g,
        a[cid] ~ dnorm(1,1),
        b[cid] ~ dexp(1),
        g ~ dexp(1),
        phi ~ dexp(1)
    ),
    data = data2,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.2 seconds.
    Total execution time: 0.9 seconds.

If you look at the graphs of the pure Poisson and the gamma-Poisson (p. 375 in the textbook) there is much less of a difference between the high and low contact trend lines. Hawaii was doing most of the work in pulling them apart, and since in gamma-Poisson model there is more expected variance, that influence is diminished.

### 2.1 Over-dispersion, entropy, and information criteria

Beta-binomial and gamma-Poisson are maximum-entropy models for the same constraints as the regular binomial and Poisson distributions. The difference is that they try to account for any unobserved heterogeneity in the populations.

You should not use PSIS or WAIC with these models unless you really know what you're doing. The reason is with the base models (binomial and Poisson) you can aggregate and disaggregate the rows (group the data) without changing the fundamental causal structure of the model. However, that is not the case with these models, since the new unobserved parameter is applied to each *row*; if we disaggregate / aggregate them, we change the population to which the parameter is being applied.

For instance, the earlier `UCBAdmit` data was grouped with total counts on each row (for each gender and department). The rows were combinations of gender + department. If we changed the data so that each row was a single person's admission (which we would want to do for WAIC since we want to estimate the accuracy of *each new admissions*) then we lose the grouping into departments which was so useful in the first place.

It doesn't seem to be a big problem in practice; in the next chapter we'll look at incorporating over-dispersion with multilevel models which let us assign heterogeneity in probabilities / rates over any level of aggregation.

## 3 Zero-Inflated Outcomes

Sometimes our observations are not the result of a pure process. Instead , they are mixtures of multiple processes. When there are multiple different causes for the same outcome, we might want to use a **mixture model**. A mixture model uses more than one simple probability distribution to model that mixture of causes; they use more than one likelihood for the same outcome.

Count variables are especially prone to mixtures. It is often the case that e.g. a zero count can represent the underlying process either not getting started at all or just actually producing a 0. For instance, if we're counting birds, a 0 count can happen because there are no birds at all or because we just failed to spot them.

### 3.1 Zero-Inflated Poisson

Let's go back to the example with the monks producing manuscripts at a certain rate. The process tends towards being Poisson. Now, let's say that on some days the monks take a vacation and drink instead of work; on those days zero manuscripts are produced. However, on other days they just happen to produce 0 manuscripts by chance. How can we estimate the number of days they spend drinking (or rather, the probability of each day being a drinking day)?

Each observed 0 can be the result of two processes.

Let $p$ be the probability of them drinking and $\lambda$ be the rate at which they complete manuscripts.

We need a likelihood function which mixes these two processes.

$$
\begin{align*}
    P(0|p,\lambda) &= P(\text{drink}|p) + P(\text{work}|p) \times P(0|\lambda) \\
                    &= p + (1-p)*\text{exp}(-\lambda)
\end{align*}
$$

since $P(y|\lambda) = \lambda^y e^{-\lambda} / y!$, $P(0|\lambda) = \lambda^0 e^{-\lambda} / 0! = e^{-\lambda}$

The likelihood of a non-zero value is

$$
\begin{align*}
    P(y|y > 0, p, \lambda) &= P(\text{drink}|p)\ast 0 + P(\text{work}|p) \ast P(y|\lambda) \\
        &= (1-p)\ast \frac{\lambda^y e^{-\lambda}}{y!}
\end{align*}
$$

Let's define ZIPoisson as the distribution above, with aprameters $p$ (probability of a zero) and $\lambda$ (mean of the Poisson) to describe the shape. Then a zero-inflated Poisson regression would look like

$$
\begin{align*}
    y_i &\sim \text{ZIPoisson}(p_i, \lambda_i) \\
    \text{logit}(p_i) &= \alpha_p + \beta_p x_i \\
    \log(\lambda_i) &= \alpha_{\lambda} + \beta_\lambda x_i
\end{align*}
$$

There are two linear models and two link functions, one for each of the constituents of the ZIPoisson. The predictors can be different between $p_i$ and $\lambda_i$ - you don't even need to use the same predictors in each one!

Now let's simulate the data and ensure that we can actually get the right data back.


```R
prob_drink <- 0.2
rate_work <- 1 # average of 1 manuscript per day

# one year
N <- 365

set.seed(365)
drink <- rbinom(N, 1, prob_drink)

# simulate maniscripts produced and total
manuscripts_worked <- rpois(N, rate_work)

y <- (1 - drink) * manuscripts_worked
```


```R
m12.3 <- ulam(
    alist(
        y ~ dzipois(p, lambda),
        logit(p) <- ap,
        log(lambda) <- al,
        ap ~ dnorm(-1.5, 1),
        al ~ dnorm(1, 0.5)
    ),
    data=list(y = y),
    chains = 4
)
precis(m12.3)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.3 seconds.
    Total execution time: 1.3 seconds.

<table class="dataframe">
<caption>A precis: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>ap</th><td>-1.305791363</td><td>0.36607506</td><td>-1.9505329</td><td>-0.8124823</td><td>1.003204</td><td>515.8134</td></tr>
	<tr><th scope=row>al</th><td> 0.002555389</td><td>0.08968448</td><td>-0.1447622</td><td> 0.1425496</td><td>1.005898</td><td>538.0696</td></tr>
</tbody>
</table>


```R
# posterior means on the natural scale
post <- extract.samples(m12.3)
post_p <- mean(inv_logit(post$ap))
post_lambda <- mean(exp(post$al))
print(post_p)
print(post_lambda)
```

    [1] 0.2194038
    [1] 1.006581


These are very close to the rates that we expect! In reality the problems will probably have lots of predictor variables that are associated with each of the parameters.

## 4 Ordered categorical outcomes

Very common to have discrete categories, but where the outcomes are ordered in some way. For instance, I might ask you to rate on a scale from 1 - 7 how much you like fish. However, be careful - with scales like this, despite the use of numbers, the difference between a 1 and 2 might be different than between a 5 and 6, for instance.

In principle, this is just a categorical prediction. However, the fact that the categories are ordered demands special consideration. What we want from the predictor variable(s) is that as they increase, the prediction should move through the categories from one to the next.

The convenctional solution is to use a **cumulative link function**. The cumulative probability of a value is the probability of that value *or any smaller value*. In the context of ordered categories, the sumulative probability of 3 is the probability of 3, 2, or 1. (Conventionally, ordered categories start at 1).

There are two steps to the explanation. The first: explain how to parameterize a distribution of outcomes on the scale of log-cumulative odds. The next is to introduce a predictor to the log-cumulative odds values, which allows the you to model associations between predictors and the outcomes while obeying the ordered nature of prediction.

### 4.1 Example: moral intuition

Data comes from experiments on the trolley problem. 

Experiments on the trolley problem has led to some principles that people seem to follow:

1. **Action Principle**: Harm caused by action is morally worse than equivalent harm caused by omission
1. **Intention principle**: Harm intended as a means to a goal is morally worse than equivalent harm foreseen as the side effect of a goal
1. **Contact principle**: Using physical contact to cause harm to a victim is morally worse than causing equivelent harm to a victim without using physical contact


```R
data(Trolley)
d <- Trolley
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 12</caption>
<thead>
	<tr><th></th><th scope=col>case</th><th scope=col>response</th><th scope=col>order</th><th scope=col>id</th><th scope=col>age</th><th scope=col>male</th><th scope=col>edu</th><th scope=col>action</th><th scope=col>intention</th><th scope=col>contact</th><th scope=col>story</th><th scope=col>action2</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>cfaqu</td><td>4</td><td> 2</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>0</td><td>1</td><td>aqu</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>cfbur</td><td>3</td><td>31</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>0</td><td>1</td><td>bur</td><td>1</td></tr>
	<tr><th scope=row>3</th><td>cfrub</td><td>4</td><td>16</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>0</td><td>1</td><td>rub</td><td>1</td></tr>
	<tr><th scope=row>4</th><td>cibox</td><td>3</td><td>32</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>1</td><td>1</td><td>box</td><td>1</td></tr>
	<tr><th scope=row>5</th><td>cibur</td><td>3</td><td> 4</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>1</td><td>1</td><td>bur</td><td>1</td></tr>
	<tr><th scope=row>6</th><td>cispe</td><td>3</td><td> 9</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>1</td><td>1</td><td>spe</td><td>1</td></tr>
</tbody>
</table>



Each row is a person's response to a story; the `response` column indicates on a scale from 1 - 7 how morally permissible they found the action taken (or not) in the story.

### 4.2 Desribing an ordered distribution with intercepts

First, let's see the distribution of outcomes


```R
ggplot(d, aes(response)) +
    geom_histogram()
```

    [1m[22m`stat_bin()` using `bins = 30`. Pick better value with `binwidth`.



    
![png](sr_chapter_12_images/output_17_1.png)
    


We want to describe this on the log-cumulative-odds scale. Why? This is the cumulative version of the **logit link** we used earlier. The logit is log-odds, and the cumulative logit is cumulative log odds.


```R
# proportion of each response
pr_k <- table(d$response) / nrow(d)

# cumsum to convert to cumulative proportions
cum_pr_k <- cumsum( pr_k )

ggplot(data.frame(x = 1:7, y = cum_pr_k), aes(x, y)) +
    geom_point() +
    geom_line()
```


    
![png](sr_chapter_12_images/output_19_0.png)
    


To move it to log-cumulative odds, we need a series of intercepts. Each intercept will be on the log-cumulative-odds scale and stand for the cumulative probability of each event. This is the application of the link function.

$$
\log \frac{P(y_i \leq k)}{1 - P(y_i \leq k)} = \alpha_k
$$


```R
logit <- function(x) log(x / (1 - x))
round(lco <- logit(cum_pr_k), 2)
```

    Warning message in log(x/(1 - x)):
    “NaNs produced”



<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>1</dt><dd>-1.92</dd><dt>2</dt><dd>-1.27</dd><dt>3</dt><dd>-0.72</dd><dt>4</dt><dd>0.25</dd><dt>5</dt><dd>0.89</dd><dt>6</dt><dd>1.77</dd><dt>7</dt><dd>NaN</dd></dl>


Note that the log odds of the top parameter are always going to be $\log(1 / 0) \to \infty$; we basically don't even need that parameter.

When we observe a $k$ and need the probability, we can use subtraction to get there:

$$
p_k = P(y_i = k) = P(y_i \leq k) - P(y_i \leq k - 1)
$$

There are lots of conventions for writing mathematical forms of the ordered logit. Here's one way:

$$
\begin{align*}
    R_i &\sim \text{Ordered-logit}(\phi_i, \kappa) & [\text{probability of data}] \\
    \phi_i &= 0 & [\text{linear model}] \\
    \kappa_k &\sim \text{Normal}(0, 1.5) & [\text{common prior for each intercept}] \\
\end{align*}
$$

Here's another, more literal way:

$$
\begin{align*}
    R_i &\sim \text{Categorical}(\vec{p}) & [\text{probability of data}] \\
    p_1 &= q_1 & [\text{probabilities of each value $k$}] \\
    p_k &= q_k - q_{k-1}\ K > k > 1 & \\
    p_K &= q - q_{k-1} \\
    \text{logit}(q_k) &= \kappa_k - \phi_i & [\text{cumulative logit link}] \\
    \phi_i &= \text{terms of the linear model} \\
    \kappa_k &\sim \text{Normal}(0, 1.5) & [\text{common prior for each intercept}] \\
\end{align*}
$$

This is a bit more complicated, but it makes it clear that the ordered categorical is just a categorical that takes a vector of probabilities.


```R
m12.4 <- ulam(
    alist(
        R ~ dordlogit(0, cutpoints),
        cutpoints ~ dnorm(0, 1.5)
    ),
    data = list(R = d$response),
    chains = 4,
    cores = 4
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 78.7 seconds.
    Total execution time: 83.0 seconds.

```R
precis(m12.4, depth=2)
```

<table class="dataframe">
<caption>A precis: 6 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>cutpoints[1]</th><td>-1.9159967</td><td>0.02953641</td><td>-1.964302</td><td>-1.8690807</td><td>1.002072</td><td>1689.377</td></tr>
	<tr><th scope=row>cutpoints[2]</th><td>-1.2659526</td><td>0.02404988</td><td>-1.304122</td><td>-1.2272884</td><td>1.000694</td><td>2092.690</td></tr>
	<tr><th scope=row>cutpoints[3]</th><td>-0.7184566</td><td>0.02138377</td><td>-0.753510</td><td>-0.6849178</td><td>1.002140</td><td>2535.823</td></tr>
	<tr><th scope=row>cutpoints[4]</th><td> 0.2475910</td><td>0.02048816</td><td> 0.215151</td><td> 0.2800441</td><td>1.000354</td><td>2791.473</td></tr>
	<tr><th scope=row>cutpoints[5]</th><td> 0.8898611</td><td>0.02187741</td><td> 0.853960</td><td> 0.9241028</td><td>1.000876</td><td>2624.785</td></tr>
	<tr><th scope=row>cutpoints[6]</th><td> 1.7692250</td><td>0.02884435</td><td> 1.722775</td><td> 1.8140920</td><td>1.000210</td><td>2270.467</td></tr>
</tbody>
</table>

```R
# get the cumulative probabilities back
round(inv_logit(coef(m12.4)), 3)
```

<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>cutpoints[1]</dt><dd>0.128</dd><dt>cutpoints[2]</dt><dd>0.22</dd><dt>cutpoints[3]</dt><dd>0.328</dd><dt>cutpoints[4]</dt><dd>0.562</dd><dt>cutpoints[5]</dt><dd>0.709</dd><dt>cutpoints[6]</dt><dd>0.854</dd></dl>



Those are roughly the same values as before; now we also have their distribution.

### 4.3 Adding predictor variables

So far we haven't accomplished much.

To include predictor variables, we define the log-cumulative-odds of each response $k$ as a sum of its intercept $\alpha_k$ and a regular linear model. Say we want to include a predictor $x$ to the model. Then we add a linear model $\phi_i = \beta x_i$, which makes the cumulative logit

$$
\begin{align*}
\log \frac{P(y_i \leq k)}{1 - P(y_i \leq k)} &= \alpha_k - \phi_i \\
\phi_i &= \beta x_i
\end{align*}
$$

Why subtract the linear model? If we decrease the log-cumulative-odds of every outcome below $k$, this shift probability mass upwards toward the higher outcome values. That means that positive values of $\beta$ means increasing $x$ also increases the mean $y$. We could certainly add $\phi$, but then $\beta > 0$ would mean that increasing $x$ decreases the mean.

For instance, let's say that we take the posterior means for `m12.4` and subtract 0.5 from each:


```R
round(pk <- dordlogit(1:7, 0, coef(m12.4)), 2)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0.13</li><li>0.09</li><li>0.11</li><li>0.23</li><li>0.15</li><li>0.15</li><li>0.15</li></ol>




```R
# average outcome
sum(pk * 1:7)
```


4.1992012542095



```R
# subtract off 0.5
round(pk <- dordlogit(1:7, 0, coef(m12.4) - 0.5), 2)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0.08</li><li>0.06</li><li>0.08</li><li>0.21</li><li>0.16</li><li>0.18</li><li>0.22</li></ol>




```R
# new average
sum(pk * 1:7)
```


4.72969448691896


By *subtracting*, we increase the mean. This way, inceasing $\beta$ implies an increase in the predictor means an increase in the outcome.

Now we can go back to the trolley problem and code in our predictors. For this model, we'll use the predictors of `action`, `intention`, and `contact`, each one corresponding for a principle outlined earlier.

Note that `contact` always implies `action`.

This gives us 6 different options:

1. No action, contact, or intention
1. Action
1. Contact
1. Intention
1. Action and intention
1. Contact and intention

The last two of these represent an interaction.

We'll use the indicator variable directly this time.

The log-cumulative-odds are now:

$$
\begin{align*}
    \log \frac{P(y_i \leq k)}{1 - P(y_i \leq k)} &= \alpha_k - \phi_i \\
    \phi_i &= \beta_A A_i + \beta_C C_i + \beta_{I,i}I_i \\
    B_{I, i} &= \beta_I + \beta_{IA}A_i + \beta_{IC}C_i \\
\end{align*}
$$

Here we've explicitly coded the interaction into a new variable $B$ to make things clearer; we could just as easily have substituted the value directly into the definition of $\phi$.

Here's the model:


```R
dat <- list(
    R = d$response,
    A = d$action,
    I = d$intention,
    C = d$contact
)

m12.5 <- ulam(
    alist(
        R ~ dordlogit(phi, cutpoints),
        phi <- bA * A + bC * C + BI * I,
        BI <- bI + bIA * A + bIC * C,
        c(bA, bI, bC, bIA, bIC) ~ dnorm(0, 0.5),
        cutpoints ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 4,
    cores = 4
)
precis(m12.5)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 169.9 seconds.
    Total execution time: 189.5 seconds.
    
    6 vector or matrix parameters hidden. Use depth=2 to show them.
    

<table class="dataframe">
<caption>A precis: 5 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>bIC</th><td>-1.2365864</td><td>0.09383004</td><td>-1.3886417</td><td>-1.0896178</td><td>1.0012167</td><td>1311.0849</td></tr>
	<tr><th scope=row>bIA</th><td>-0.4318807</td><td>0.07841415</td><td>-0.5569462</td><td>-0.3070851</td><td>1.0000339</td><td>1362.9769</td></tr>
	<tr><th scope=row>bC</th><td>-0.3414661</td><td>0.06515598</td><td>-0.4472851</td><td>-0.2396374</td><td>1.0024492</td><td>1216.8330</td></tr>
	<tr><th scope=row>bI</th><td>-0.2931201</td><td>0.05689291</td><td>-0.3839079</td><td>-0.2031280</td><td>1.0004210</td><td>1041.3692</td></tr>
	<tr><th scope=row>bA</th><td>-0.4728817</td><td>0.05304754</td><td>-0.5566313</td><td>-0.3887871</td><td>0.9998076</td><td> 982.4266</td></tr>
</tbody>
</table>



Note that they're all negative - each of these factors decreases the acceptability of the story / action


```R
par(bg='white')
plot(precis(m12.5), xlim=c(-1.4, 0))
```

    6 vector or matrix parameters hidden. Use depth=2 to show them.

    
![png](sr_chapter_12_images/output_35_1.png)
    

This would be easier to see if we could plot the posterior predictions. However, that is tricky, since each prediction is really a vector of probabilities - one for each possible outcome value. As the predictor variables change, so do the *entire vector* of outputs.

One common way to do this is to have the horizontal axis for a predictor variable and the vertical axis for cumulative probability. Then we can plot a curve for each response value as it changes across values of the predictor variable. If we plot a curve for each response value, we'll end up with a distribution of responses as it changes across values of the predictor variable.


```R
par(bg='white')
plot(NULL, type="n", xlab="intention", ylab="probability", xlim=c(0, 1), ylim=c(0, 1), xaxp=c(0, 1, 1), yaxp=c(0, 1, 2))

# set up the data list that contains the different combinations of predictors.
kA <- 0 # action
kC <- 0 # contact
kI <- 0:1 # intention
pdat <- data.frame(A = kA, C = kC, I = kI)

# pass it to link to get the phi samples
phi <- link(m12.5, data = pdat)$phi

# loop over the first 50 samples and plot their predictions across values of intention
# Trick is to use `pordlogit` to compute the cumulative probability use the sampes in phi and the cutpoints
post <- extract.samples(m12.5)
for (s in 1:50) {
    pk <- pordlogit(1:6, phi[s, ], post$cutpoints[s,])
    for (i in 1:6) {
        lines(kI, pk[, i], col=grau(0.1))
    }
}
```


    
![png](sr_chapter_12_images/output_37_0.png)
    


By changing the values of kA and kC, we could see the effect of changing I in each of these scenarios. 

We can also create a histogram of the outcomes by using `sim` to simulate posterior outcomes.


```R
par(bg='white')
kA <- 0
kC <- 1
kI <- 0:1
pdat <- data.frame(A = kA, C = kC, I = kI)
s <- sim(m12.5, data = pdat)
simplehist(s, xlab = "Response")
```


    
![png](sr_chapter_12_images/output_39_0.png)
    


Black lines are when I = 0, blue is I = 1. Essentially, the story here is that going from no interaction -> interaction increases the chances of a low response and decreases the chances of a high response.

## 5 Ordered categorical predictors

So far we've looked at categorical *outcomes*. What if the predictors are some sort of ordered category? In principle we could pretend that they are linear, but just like we didn't want to do that for the outcomes it's an equally bad idea here.

The Trolley dataset has an example: education level.


```R
levels(d$edu)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'Bachelor\'s Degree'</li><li>'Elementary School'</li><li>'Graduate Degree'</li><li>'High School Graduate'</li><li>'Master\'s Degree'</li><li>'Middle School'</li><li>'Some College'</li><li>'Some High School'</li></ol>



There are 8 levels here. Unfortunately they're not in the correct order, so we need to fix that.


```R
# This is actually us replacing the text of the level with the numerical representation
edu_levels <- c(6, 1, 8, 4, 7, 2, 5, 3)
d$edu_new <- edu_levels[d$edu]
d$edu_new
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>7</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>⋯</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>5</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>4</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>6</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li><li>8</li></ol>



The idea behind ordered predictor variables is that each step up comes with its own incremental / marginal effect on the outcome. That means that we want to infer what each of those effects is. With 8 levels, we need 7 parameters (1 -> 2, 2 -> 3, &c.).

The first parameter (elementary school) will be absorbed into the intercept.

Our linear model is starting to look as follows:

$$
\phi_i = \delta_1 + \text{other stuff}
$$

Here $\delta_1$ is the effect of completing middle school (middle school -> some high school), and 'other stuff' is all of the other effects that we haven't dealt with yet.

A different individual completes the next level (Some High School), so their model would be

$$
\phi_i = \delta_1 + \delta_2 + \text{other stuff}
$$

Someone with the highest level, a graduate degree, woud have a linear model like

$$
\sum_{j=1}^{7}\delta_i + \text{other stuff}
$$

It's convenient for later interpretation if we call the maximum sum and ordinary coefficient like $\beta_E$ and then let all of the $\delta$ parameters be fractions of this maximum sum. If we include a dummy $\delta_0 = 0$, then our model is

$$
\phi_i = \sum_{j=0}^{E_i-1}\delta_j + \text{other stuff}
$$

Where $E_i$ is the maximum educational attainment of individual $i$. Note that the sum of $\delta_j$ is 1. Also, if someone has the lowest education attainment, then $\beta_E$ doesn't appear in the model because it will only be multiplying the dummy $\delta_0 = 0$.

Defining it in this way also helps us to define priors for $\beta_E$ and the rest. If we expect the levels to be equally spaced on the underlying linear scale, then we can give them all the same prior. We can separately set a prior on $\beta_E$ to determine the total effect of education (or whatever).

Let's see what that model will look like for us:

$$
\begin{align*}
R_i &\sim \text{Ordered-logit}(\phi_i, \kappa) \\
\phi_i &= \beta_E \sum_{j=0}^{E_i - 1}\delta_j + \beta_A A_i + \beta_I I_i + \beta_C C_i \\
\end{align*}
$$

Now we need a whole bunch of priors. The priors for the cutpoints are on the logit scale, so we'll use our regularizing priors with standard deviation 1.5. The slopes get narrower priors - each of these is the log-odds difference

$$
\begin{align*}
\kappa &\sim \text{Normal}(0, 1.5) \\
\beta_A, \beta_I, \beta_C, \beta_E &\sim \text{Normal}(0, 1) \\
\delta &\sim \text{Dirichlet}(\alpha)
\end{align*}
$$

This last line is new - the prior for $\delta$ is from the [[Dirichlet Distribution]]. This is the multivariate version of the [[Beta Distribution]]. This distribution is for a vector of values, all between 0 and one, which sum to 1. The **beta** is the distribution for two values; the **Dirichlet** extends this to any number. Like the **beta**, it is parameterized by 'pseudo-counts' for each possibility. In the beta they were the number of successes and failures ($\alpha$ and $\beta$); for the **Dirichlet** it is just a vector of counts (successes, presumably).


```R
set.seed(1805)
delta <- rdirichlet(10, alpha = rep(2, 7))
delta
```


<table class="dataframe">
<caption>A matrix: 10 × 7 of type dbl</caption>
<tbody>
	<tr><td>0.10531402</td><td>0.03999053</td><td>0.07958541</td><td>0.12153233</td><td>0.24473313</td><td>0.13780806</td><td>0.27103653</td></tr>
	<tr><td>0.25042007</td><td>0.08914921</td><td>0.08122755</td><td>0.16279318</td><td>0.09614976</td><td>0.12892247</td><td>0.19133775</td></tr>
	<tr><td>0.19169161</td><td>0.24096413</td><td>0.01190757</td><td>0.06676158</td><td>0.26010974</td><td>0.19841979</td><td>0.03014558</td></tr>
	<tr><td>0.12405712</td><td>0.11851318</td><td>0.33153891</td><td>0.03516935</td><td>0.13764852</td><td>0.12844491</td><td>0.12462801</td></tr>
	<tr><td>0.08774940</td><td>0.03240889</td><td>0.15166692</td><td>0.24049639</td><td>0.06173455</td><td>0.23906111</td><td>0.18688275</td></tr>
	<tr><td>0.13462030</td><td>0.05974400</td><td>0.18765413</td><td>0.22280044</td><td>0.10288096</td><td>0.07458804</td><td>0.21771214</td></tr>
	<tr><td>0.10334697</td><td>0.09848294</td><td>0.03551862</td><td>0.23140054</td><td>0.19133775</td><td>0.25018561</td><td>0.08972758</td></tr>
	<tr><td>0.23671078</td><td>0.17581052</td><td>0.04317941</td><td>0.12924604</td><td>0.21926430</td><td>0.07135114</td><td>0.12443780</td></tr>
	<tr><td>0.07347475</td><td>0.21009599</td><td>0.17633836</td><td>0.19569642</td><td>0.22037361</td><td>0.07481659</td><td>0.04920429</td></tr>
	<tr><td>0.33541220</td><td>0.06823140</td><td>0.05123645</td><td>0.26034135</td><td>0.16708028</td><td>0.02614977</td><td>0.09154854</td></tr>
</tbody>
</table>

We end up with 10 vectors of 7 probabilities, each summing to 1


```R
plot_df <- data.frame(x = integer(), p = numeric(), group = integer())
for (i in 1:nrow(delta)) {
    plot_df <- rbind(plot_df, data.frame(
        x = 1:7,
        p = delta[i, ],
        group = i
    ))
}
head(plot_df)
```


<table class="dataframe">
<caption>A data.frame: 6 × 3</caption>
<thead>
	<tr><th></th><th scope=col>x</th><th scope=col>p</th><th scope=col>group</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>0.10531402</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>2</td><td>0.03999053</td><td>1</td></tr>
	<tr><th scope=row>3</th><td>3</td><td>0.07958541</td><td>1</td></tr>
	<tr><th scope=row>4</th><td>4</td><td>0.12153233</td><td>1</td></tr>
	<tr><th scope=row>5</th><td>5</td><td>0.24473313</td><td>1</td></tr>
	<tr><th scope=row>6</th><td>6</td><td>0.13780806</td><td>1</td></tr>
</tbody>
</table>




```R
ggplot(plot_df, aes(x, p, group = group)) +
    geom_point() +
    geom_line(aes(linewidth = group == 3)) 
```

    Warning message:
    “[1m[22mUsing [32mlinewidth[39m for a discrete variable is not advised.”



    
![png](sr_chapter_12_images/output_49_1.png)
    


We also need to deal with the fact that $\delta_0 = 0$.


```R
dat <- list(
    R = d$response,
    action = d$action,
    intention = d$intention,
    contact = d$contact,
    E = as.integer(d$edu_new), # this is an index
    alpha = rep(2, 7) # delta prior
)

m12.6 <- ulam(
    alist(
        R ~ ordered_logistic(phi, kappa),
        phi <- bE * sum(delta_j[1:E]) + bA * action + bI * intention + bC * contact,
        kappa ~ normal(0, 1.5),
        c(bA, bI, bC, bE) ~ normal(0, 1),
        vector[8]: delta_j <<- append_row(0, delta),
        simplex[7]: delta ~ dirichlet(alpha)
    ),
    data = dat,
    chains = 4,
    cores = 4
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 27719.9 seconds.
    Total execution time: 30491.7 seconds.

Few notes:
- we're passing in the prior as data; that's fine to do
- Summing over the first $E$ values of $\delta_d$ 
- We build the `delta_j` vector using the `append_row` method
    - This is from [[Stan]], not a native R one
- We call the resultant vector a [[Simplex]]. A simplex is another name for a vector whose values sum to some given value (usually 1). Again, this is a Stan object. Putting the `simplex[7]` in front tells Stan that the variable is a simplex of length 7
- This model takes forever to run (!)


```R
precis(m12.6, depth = 2, omit = "kappa")
```


<table class="dataframe">
<caption>A precis: 17 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>kappa[1]</th><td>-3.07675731</td><td>0.15199411</td><td>-3.331083050</td><td>-2.86354015</td><td>1.0099402</td><td> 831.6626</td></tr>
	<tr><th scope=row>kappa[2]</th><td>-2.39441591</td><td>0.14928560</td><td>-2.648600150</td><td>-2.18863925</td><td>1.0094658</td><td> 820.5333</td></tr>
	<tr><th scope=row>kappa[3]</th><td>-1.81180439</td><td>0.14832242</td><td>-2.063176300</td><td>-1.60642865</td><td>1.0092105</td><td> 824.1459</td></tr>
	<tr><th scope=row>kappa[4]</th><td>-0.79017528</td><td>0.14770321</td><td>-1.037327800</td><td>-0.58142634</td><td>1.0089861</td><td> 844.8868</td></tr>
	<tr><th scope=row>kappa[5]</th><td>-0.12218845</td><td>0.14759632</td><td>-0.369906910</td><td> 0.08451394</td><td>1.0091870</td><td> 861.9233</td></tr>
	<tr><th scope=row>kappa[6]</th><td> 0.78474039</td><td>0.14848037</td><td> 0.532156745</td><td> 0.99407491</td><td>1.0094681</td><td> 860.1058</td></tr>
	<tr><th scope=row>bE</th><td>-0.31517534</td><td>0.16879583</td><td>-0.595084645</td><td>-0.07756135</td><td>1.0093685</td><td> 883.4559</td></tr>
	<tr><th scope=row>bC</th><td>-0.95537364</td><td>0.05130479</td><td>-1.038319350</td><td>-0.87494732</td><td>1.0015069</td><td>2234.0763</td></tr>
	<tr><th scope=row>bI</th><td>-0.71638951</td><td>0.03767764</td><td>-0.774022035</td><td>-0.65618056</td><td>0.9998169</td><td>3381.0185</td></tr>
	<tr><th scope=row>bA</th><td>-0.70329414</td><td>0.04235110</td><td>-0.770940315</td><td>-0.63635313</td><td>1.0030986</td><td>2414.1654</td></tr>
	<tr><th scope=row>delta[1]</th><td> 0.22711858</td><td>0.13527568</td><td> 0.052594735</td><td> 0.47145300</td><td>1.0073370</td><td>1212.4423</td></tr>
	<tr><th scope=row>delta[2]</th><td> 0.14209005</td><td>0.08953907</td><td> 0.031159830</td><td> 0.31182948</td><td>1.0019117</td><td>2250.7237</td></tr>
	<tr><th scope=row>delta[3]</th><td> 0.19545434</td><td>0.11223559</td><td> 0.047357984</td><td> 0.40200492</td><td>1.0014951</td><td>2134.1301</td></tr>
	<tr><th scope=row>delta[4]</th><td> 0.17096330</td><td>0.09547568</td><td> 0.040300925</td><td> 0.34332163</td><td>1.0009113</td><td>2519.9967</td></tr>
	<tr><th scope=row>delta[5]</th><td> 0.04190406</td><td>0.04784824</td><td> 0.005546966</td><td> 0.11297132</td><td>1.0039897</td><td>1690.5523</td></tr>
	<tr><th scope=row>delta[6]</th><td> 0.09905737</td><td>0.06354878</td><td> 0.022223182</td><td> 0.21400941</td><td>1.0015995</td><td>2256.4041</td></tr>
	<tr><th scope=row>delta[7]</th><td> 0.12341232</td><td>0.07420887</td><td> 0.025958869</td><td> 0.25620603</td><td>1.0009942</td><td>2373.4611</td></tr>
</tbody>
</table>



The overall effect of `bE` is negative - more educated individuals disapprove of everything. It's not as powerful an effect as e.g. adding action (-0.3 v. -0.7).

To see what's going on with the `delta` parameters, we'll have to look at them as a multivariate distribution.


```R
par(bg = 'white')
options(repr.plot.width = 16, repr.plot.height = 10)
delta_labels <- c("Elem", "MidSch", "SHS", "SCol", "Bach", "Mast", "Grad")
pairs(m12.6, pars = "delta", labels = delta_labels)
```

    
![png](sr_chapter_12_images/output_55_2.png)
    

Not sure what's going on here - the `pars` parameter doesn't actually seem to do anything.

It's interesting to compare the above model, where the scale isn't linear, to one where we assume that education is linear


```R
dat$edu_norm <- normalize(d$edu_new)
m12.7 <- ulam(
    alist(
        R ~ ordered_logistic(mu, cutpoints),
        mu <- bE * edu_norm + bA * action + bI * intention + bC * contact,
        c(bA, bI, bC, bE) ~ normal(0, 1),
        cutpoints ~ normal(0, 1.5)
    ),
    data = dat,
    chains = 4,
    cores = 4
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 229.3 seconds.
    Total execution time: 240.2 seconds.

```R
precis(m12.7)
```

    6 vector or matrix parameters hidden. Use depth=2 to show them.
    

<table class="dataframe">
<caption>A precis: 4 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>bE</th><td>-0.09921081</td><td>0.09069224</td><td>-0.2453261</td><td> 0.0438181</td><td>1.0006583</td><td>1685.959</td></tr>
	<tr><th scope=row>bC</th><td>-0.95640186</td><td>0.05013702</td><td>-1.0352065</td><td>-0.8771060</td><td>1.0040052</td><td>2191.218</td></tr>
	<tr><th scope=row>bI</th><td>-0.71759919</td><td>0.03717072</td><td>-0.7777304</td><td>-0.6595751</td><td>1.0000563</td><td>2172.068</td></tr>
	<tr><th scope=row>bA</th><td>-0.70392592</td><td>0.04268818</td><td>-0.7700486</td><td>-0.6356098</td><td>0.9999081</td><td>2401.170</td></tr>
</tbody>
</table>



This model thinks that education is not as strongly associated with rating. This could be because the effect actually isn't linear; different levels may have different incremental associations.

This model has been fine, but there's something that should worry us: age. Almost surely age is associted with education; you probably don't have a Master's degree if you're 12 years old. It is plausible that there is a backdoor from age to through age to rating. We'll investigate this later.

## 6 Practice

#### 6.1.1 12E1 What is the difference between an ordered and unordered categorical variable? Define then give an example of each.

An unordered categorical variable does not reflect any underlying increase or decrease in an attribute of interest, whereas an ordered one does. For instance, when looking at a forest, the different species would represent an unordered categorical variable ('elm', 'oak', 'dandelion') whereas their height position in the canopy would be an ordered categorical variable ('canopy', 'shrub layer', 'ground cover').

#### 6.1.2 12E2 What kind of link function does an ordered logistic regression employ? How does it differ from an ordinary logit link?

An ordered logistic regression uses a **cumulative link function**. This is similar to the ordinary logit function, but where the logit function is the log-odds of an event orccurring, the cumulative link function is the log-cumulative-odds - that is, the log of the odds of that event or something lower on the scale occurring.

#### 6.1.3 12E3 When count data are zero-inflated, using a model that ignores zero-inflation will tend to introduce what kinds of inferential cost?

Ignoring zero-inflation will result in an estimate of the true rate that is lower than the true value; the zero-inflation will drag the estimated value towards zero.

#### 6.1.4 12E4 Over-dispersion is common in count data. Give an example of a natural process that might produce over-dispersed counts. Can you also give an example of a process that might produce *under*-dispersed counts?

A process that might involve over-dispersed data is presence / absence data of fossil organisms. We might find no fossils there because they generally weren't in that area, or because they were and just failed to fossilize.

A process that might involve under-dispersed counts is the occurrence of failing students in a class. Generally if someone 'just' fails (maybe by only a few percentage points), the professor will give them the benefit of the doubt and boost them a few points to avoid an outright failure. In that case fewer students would be failing than we would expect from the distribution of the rest of the grades.

#### 6.1.5 12M1 At a certain university, employees are annually rated from 1 to 4 on their productivity, with 1 being least productive and 4 being most. In a certain department at this university, the number of employees receiving each rating from (from 1 - 4): 12, 36, 7, 41. Compute the log cumulative odds of each rating.


```R
odds <- function(p) p / (1 - p)
log_odds <- function(p) log(odds(p))
cumulative_log_odds <- function(counts, i) log_odds(sum(counts[1:i]) / sum(counts))

ratings <- c(12, 36, 7, 41)
for (i in 1:length(ratings)) {
    print(paste(i, cumulative_log_odds(ratings, i)))
}
```

    [1] "1 -1.94591014905531"
    [1] "2 0"
    [1] "3 0.293761118528163"
    [1] "4 Inf"


#### 6.1.6 12M2 Make a version of Figure 12.5 for the employee data rating given above


```R
plot_df <- data.frame(
    rating = 1:4,
    count = ratings,
    proportion = ratings / sum(ratings),
    cumulative = cumsum(ratings / sum(ratings)),
    prev = cumsum(ratings / sum(ratings)) - ratings / sum(ratings)
)

jitter_offset = 0.05
ggplot(plot_df, aes(rating)) +
    geom_line(aes(y = cumulative)) +
    geom_segment(mapping = aes(y = 0, xend = rating, yend = cumulative)) +
    geom_point(aes(y = cumulative), shape = 21, size = 5, fill = 'white') +
    geom_segment(mapping = aes(x = rating + jitter_offset, y = prev, xend = rating + jitter_offset, yend = cumulative), colour = 'blue', linewidth = 2)
```


    
![png](sr_chapter_12_images/output_68_0.png)
    


#### 6.1.7 12M3 Can you modify the derivation of the zero-inflated Poisson distribution `ZIPoisson` from this chapter to construct a zero-inflated binomial distribution?

Say you have some sort of situation distributed binomially with probability $p$ and count $n$. You also have some sort of condition such that, with probability $p_0$, you will see 0 of whatever you're counting.

For the binomial, the probability distribution is

$$
P_b(x) = \binom{n}{x}p^x(1-p)^{n - x}
$$

and the probability that you get zero events from the binomial is

$$
P_b(0) = p^0 \binom{n}{0}(1 - p)^{n - 0} = (1-p)^n
$$

If we see zero events, that can either happen because the blocking event occurred (probability $p_0$) or because it didn't and we got a natural zero from the binomial:

$$
P(0) = p_0 + (1 - p_0)(1-p)^n
$$

And the probability that we see $x > 0$ events is the probability that the blocking event did not happen and also that the binomial produced $x$ events:

$$
P(x|x > 0) = (1-p_0) * \binom{n}{x} p^x (1 - p)^{n - x}
$$

So then the full probability is

$$
P(x) = \begin{cases}
p_0 + (1 - p_0)(1-p)^n & x = 0 \\
(1-p_0) \ast \binom{n}{x} p^x (1 - p)^{n - x} & x > 0 \\
\end{cases}
$$

#### 6.1.8 12H1 In 2014, a paper was published titled "Female hurricanes are deadlier than male hurricanes". The paper claimed that hurricanes with female names have caused greater loss of life; the explanation given was that people subconsciously rate female hurricanes as less dangerous and so are less likely to evacuate. Statisticians severely criticized the paper after publication. Here, we'll consider the full data from the paper and consider the hypothesis that hurricanes with female names are deadlier. Load the data with
```R
library(rethinking)
data(Hurricanes)
```

Acquaint yourself with the columns by using `?Hurricanes`. In this problem, you'll focus on predicting `deaths` using `femininity` of each hurricane's name. Fit and interpret the simplest possible model, a Poisson model of `deaths` using `femininity` as a predictor. You can use `quap` or `ulam`. Compare the model to an intercept-only Poisson model of `deaths`. How strong is the association between femininity of a name and deaths? Which storms does the model fit (retrodict) well? Which storms does it fit poorly?


```R
data(Hurricanes)
d <- Hurricanes
head(d)
summary(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 8</caption>
<thead>
	<tr><th></th><th scope=col>name</th><th scope=col>year</th><th scope=col>deaths</th><th scope=col>category</th><th scope=col>min_pressure</th><th scope=col>damage_norm</th><th scope=col>female</th><th scope=col>femininity</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Easy    </td><td>1950</td><td> 2</td><td>3</td><td>960</td><td> 1590</td><td>1</td><td>6.77778</td></tr>
	<tr><th scope=row>2</th><td>King    </td><td>1950</td><td> 4</td><td>3</td><td>955</td><td> 5350</td><td>0</td><td>1.38889</td></tr>
	<tr><th scope=row>3</th><td>Able    </td><td>1952</td><td> 3</td><td>1</td><td>985</td><td>  150</td><td>0</td><td>3.83333</td></tr>
	<tr><th scope=row>4</th><td>Barbara </td><td>1953</td><td> 1</td><td>1</td><td>987</td><td>   58</td><td>1</td><td>9.83333</td></tr>
	<tr><th scope=row>5</th><td>Florence</td><td>1953</td><td> 0</td><td>1</td><td>985</td><td>   15</td><td>1</td><td>8.33333</td></tr>
	<tr><th scope=row>6</th><td>Carol   </td><td>1954</td><td>60</td><td>3</td><td>960</td><td>19321</td><td>1</td><td>8.11111</td></tr>
</tbody>
</table>




           name         year          deaths          category      min_pressure   
     Bob     : 3   Min.   :1950   Min.   :  0.00   Min.   :1.000   Min.   : 909.0  
     Bonnie  : 2   1st Qu.:1965   1st Qu.:  2.00   1st Qu.:1.000   1st Qu.: 950.0  
     Charley : 2   Median :1985   Median :  5.00   Median :2.000   Median : 964.0  
     Cindy   : 2   Mean   :1982   Mean   : 20.65   Mean   :2.087   Mean   : 964.9  
     Danny   : 2   3rd Qu.:1999   3rd Qu.: 20.25   3rd Qu.:3.000   3rd Qu.: 982.2  
     Florence: 2   Max.   :2012   Max.   :256.00   Max.   :5.000   Max.   :1003.0  
     (Other) :79                                                                   
      damage_norm        female         femininity    
     Min.   :    1   Min.   :0.0000   Min.   : 1.056  
     1st Qu.:  245   1st Qu.:0.0000   1st Qu.: 2.667  
     Median : 1650   Median :1.0000   Median : 8.500  
     Mean   : 7270   Mean   :0.6739   Mean   : 6.781  
     3rd Qu.: 8162   3rd Qu.:1.0000   3rd Qu.: 9.389  
     Max.   :75000   Max.   :1.0000   Max.   :10.444  
                                                      



```R
?Hurricanes
```

    Hurricanes             package:rethinking              R Documentation
    
    _H_u_r_r_i_c_a_n_e _f_a_t_a_l_i_t_i_e_s _a_n_d _g_e_n_d_e_r _o_f _n_a_m_e_s
    
    _D_e_s_c_r_i_p_t_i_o_n:
    
         Data used in Jung et al 2014 analysis of effect of gender of name
         on hurricane fatalities. Note that hurricanes Katrina (2005) and
         Audrey (1957) were removed from the data.
    
    _U_s_a_g_e:
    
         data(Hurricanes)
         
    _F_o_r_m_a_t:
    
           1. name : Given name of hurricane
    
           2. year : Year of hurricane
    
           3. deaths : number of deaths
    
           4. category : Severity code for storm
    
           5. min_pressure : Minimum pressure, a measure of storm strength;
              low is stronger
    
           6. damage_norm : Normalized estimate of damage in dollars
    
           7. female : Indicator variable for female name
    
           8. femininity : 1-11 scale from totally masculine (1) to totally
              feminine (11) for name. Average of 9 scores from 9 raters.
    
    _R_e_f_e_r_e_n_c_e_s:
    
         Jung et al. 2014. Female hurricanes are deadlier than male
         hurricanes. PNAS.



```R
ggplot(d, aes(year, min_pressure)) +
    geom_point(aes(size = deaths))
```


    
![png](sr_chapter_12_images/output_73_0.png)
    



```R
p <- ggplot(d, aes(femininity, deaths)) +
    geom_point()
print(p)
```


    
![png](sr_chapter_12_images/output_74_0.png)
    



```R
# priors
# initial model is deaths ~ Poisson(lambda), log(lambda) ~ alpha

# figuring out lambda - find the mean number of deaths
mean_deaths <- mean(d$deaths)
log_mean_deaths <- log(mean_deaths)
sd_deaths <- sd(d$deaths)

ALPHA_PRIOR_MEAN <- log_mean_deaths
ALPHA_PRIOR_SD <- 2

NUM_SAMPLES <- 1e1
alpha <- rnorm(NUM_SAMPLES, ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD)
lambda <- exp(alpha)
for (i in 1:NUM_SAMPLES) {
    predicted_deaths <- rpois(1, lambda[i])
    femininities <- seq(1, 11, length.out = 25)
    p <- p + geom_line(data = data.frame(femininity = femininities, deaths = predicted_deaths))
}
print(p)
```


    
![png](sr_chapter_12_images/output_75_0.png)
    


This looks slightly too wide but generally in the right ballpark. Now for the simple linear model!


```R
p <- ggplot(d, aes(femininity, deaths)) +
    geom_point()

BETA_F_PRIOR_MEAN <- 0
BETA_F_PRIOR_SD <- 0.05

NUM_SAMPLES <- 1e1
alpha <- rnorm(NUM_SAMPLES, ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD)
beta_f <- rnorm(NUM_SAMPLES, BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD)
femininities <- seq(1, 11, length.out = 25)
for (i in 1:NUM_SAMPLES) {
    # just go with the mean for now
    log_predicted_deaths <- alpha[i] + beta_f[i] * femininities
    predicted_deaths <- exp(log_predicted_deaths)
    p <- p + geom_line(data = data.frame(femininity = femininities, deaths = predicted_deaths))
}
print(p)

```


    
![png](sr_chapter_12_images/output_77_0.png)
    



```R
# intercept-only Poisson
hurricane_data <- list(
    deaths = d$deaths,
    femininity = d$femininity,
    ALPHA_PRIOR_MEAN = ALPHA_PRIOR_MEAN,
    ALPHA_PRIOR_SD = ALPHA_PRIOR_SD
)
m12h1.intercept <- ulam(
    alist(
        deaths ~ dpois(lambda),
        log(lambda) <- alpha,
        alpha ~ dnorm(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD)
    ),
    data = hurricane_data,
    log_lik = TRUE
)

pre <- precis(m12h1.intercept)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    
    Chain 1 finished in 0.0 seconds.



```R
par(bg = 'white')
print(pre)
plot(pre)
```

              mean         sd     5.5%    94.5%     rhat ess_bulk
    alpha 3.029668 0.02216972 2.994572 3.062744 1.009995 150.8799



    
![png](sr_chapter_12_images/output_79_1.png)
    



```R
# intercept and slope Poisson
hurricane_data <- list(
    deaths = d$deaths,
    femininity = d$femininity,
    ALPHA_PRIOR_MEAN = ALPHA_PRIOR_MEAN,
    ALPHA_PRIOR_SD = ALPHA_PRIOR_SD,
    BETA_F_PRIOR_MEAN = BETA_F_PRIOR_MEAN,
    BETA_F_PRIOR_SD = BETA_F_PRIOR_SD
)
m12h1.intercept_slope <- ulam(
    alist(
        deaths ~ dpois(lambda),
        log(lambda) <- alpha + bf * femininity,
        alpha ~ dnorm(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD),
        bf ~ dnorm(BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD)
    ),
    data = hurricane_data,
    log_lik = TRUE
)
pre.slope <- precis(m12h1.intercept_slope)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    
    Chain 1 finished in 0.1 seconds.



```R
par(bg = 'white')
print(pre.slope)
plot(pre.slope)
```

                mean         sd       5.5%      94.5%     rhat ess_bulk
    alpha 2.50723104 0.06217910 2.40662470 2.60359315 1.014877 93.25997
    bf    0.07272826 0.00761685 0.06109565 0.08517655 1.012791 90.85604



    
![png](sr_chapter_12_images/output_81_1.png)
    


So according to this, there is a positive relation between the femininity of the names and the deaths ($\beta_f > 0$). Now let's compare the models: 


```R
compare(m12h1.intercept, m12h1.intercept_slope)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m12h1.intercept_slope</th><td>4432.122</td><td>1008.63</td><td> 0.00000</td><td>      NA</td><td>141.11763</td><td>0.997673315</td></tr>
	<tr><th scope=row>m12h1.intercept</th><td>4444.244</td><td>1071.62</td><td>12.12196</td><td>135.2889</td><td> 74.68958</td><td>0.002326685</td></tr>
</tbody>
</table>



So the model with the intercept does significantly better than the one without.

Now let's see which hurricanes this does a good job predicting (or not):


```R
femininity_sims <- list(femininity = seq(1, 11, length.out = 100))
simulated_deaths <- link(m12h1.intercept_slope, data = femininity_sims)
simulated_deaths
```


<table class="dataframe">
<caption>A matrix: 500 × 100 of type dbl</caption>
<tbody>
	<tr><td>14.00478</td><td>14.10134</td><td>14.19858</td><td>14.29648</td><td>14.39506</td><td>14.49431</td><td>14.59426</td><td>14.69489</td><td>14.79621</td><td>14.89824</td><td>⋯</td><td>25.99334</td><td>26.17257</td><td>26.35304</td><td>26.53475</td><td>26.71771</td><td>26.90194</td><td>27.08743</td><td>27.27421</td><td>27.46227</td><td>27.65163</td></tr>
	<tr><td>13.11501</td><td>13.20847</td><td>13.30259</td><td>13.39738</td><td>13.49285</td><td>13.58900</td><td>13.68584</td><td>13.78337</td><td>13.88159</td><td>13.98051</td><td>⋯</td><td>24.84915</td><td>25.02623</td><td>25.20457</td><td>25.38417</td><td>25.56506</td><td>25.74724</td><td>25.93071</td><td>26.11550</td><td>26.30159</td><td>26.48902</td></tr>
</tbody>
</table>

```R
mean_simulated_deaths <- apply(simulated_deaths, 2, mean)
bounds_simulated_deaths <- apply(simulated_deaths, 2, function(col) PI(col, prob = 0.95))
simulated_df <- data.frame(
    femininity = femininity_sims$femininity,
    mean = mean_simulated_deaths,
    lower = bounds_simulated_deaths[1, ],
    upper = bounds_simulated_deaths[2, ]
)
```


```R
base_plot <- ggplot(d, aes(femininity)) +
    geom_point(aes(y = deaths))
base_plot +
    geom_line(data = simulated_df, mapping = aes(femininity, mean)) +
    geom_ribbon(data = simulated_df, mapping = aes(femininity, ymin = lower, ymax = upper), alpha = 0.2)
```


    
![png](sr_chapter_12_images/output_87_0.png)
    


So actually this does a pretty terrible job at predicting anything. The model is convinced that everything is going to live in a very narrow band, when in reality there is a tonne more variation. My guess is that some of the hurricanes with large death counts are driving the slight positive trend that we see.

#### 6.1.9 12H2 Counts are nearly always over-dispersed relative to Poisson. So fit a gamma-Poisson (aka negative-binomial) model to predict `deaths` using `femininity`. Show that the over-dispersed model no longer shows as precise a positive association between femininity and deaths, with an 89% interval that overlaps zero. Can you explain why the association diminished in strength?


```R
m12h2 <- ulam(
    alist(
        deaths ~ dgampois(lambda, phi),
        log(lambda) <- alpha + bf * femininity,
        alpha ~ dnorm(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD),
        bf ~ dnorm(BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD),
        phi ~ dexp(1)
    ),
    data = hurricane_data,
    log_lik = TRUE
)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    Chain 1 finished in 0.2 seconds.

```R
precis(m12h2)
par(bg = 'white')
plot(precis(m12h2))
```

<table class="dataframe">
<caption>A precis: 3 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>alpha</th><td>2.78866568</td><td>0.30172792</td><td> 2.3405157</td><td>3.28875375</td><td>0.9991627</td><td>105.5894</td></tr>
	<tr><th scope=row>bf</th><td>0.03540821</td><td>0.03559844</td><td>-0.0214341</td><td>0.09311813</td><td>0.9993022</td><td>104.5446</td></tr>
	<tr><th scope=row>phi</th><td>0.44672699</td><td>0.06159693</td><td> 0.3615986</td><td>0.55373324</td><td>0.9997913</td><td>311.8078</td></tr>
</tbody>
</table>

    
![png](sr_chapter_12_images/output_91_1.png)
    



```R

femininity_sims <- list(femininity = seq(1, 11, length.out = 100))
simulated_deaths <- link(m12h2, data = femininity_sims)
mean_simulated_deaths <- apply(simulated_deaths, 2, mean)
bounds_simulated_deaths <- apply(simulated_deaths, 2, function(col) PI(col, prob = 0.95))
simulated_df <- data.frame(
    femininity = femininity_sims$femininity,
    mean = mean_simulated_deaths,
    lower = bounds_simulated_deaths[1, ],
    upper = bounds_simulated_deaths[2, ]
)
base_plot +
    geom_line(data = simulated_df, mapping = aes(femininity, mean)) +
    geom_ribbon(data = simulated_df, mapping = aes(femininity, ymin = lower, ymax = upper), alpha = 0.2)
```


    
![png](sr_chapter_12_images/output_92_0.png)
    


The over-dispersed model produces a coefficient for femininity that overlaps zero; it is less certain of the effect of the femininity of the name on the deadliness of the hurricane. Looking at the above plot, we can also see that the prediction line is more flat and has wider bars; the model as a whole is less certain about the outcomes.

My suspicion is that some of the effect of femininity that was reported by the previous model was being unduly influenced by the very high-death feminine hurricanes; since the gamma-Poisson model expects more variabiliy, those outliers are less influential.

#### 6.1.10 12H3 In the data, there are two measures of a hurricane's potential to cause death: `damage_norm` and `min_pressure`. Consult `?Hurricanes` for their meaning. It makes some sense to imagine that femininity of a name matters more when the hurricane is itself deadly. This implies an interaction between `femininity` and either both or one of `damage_norm` and `min_pressure`. Fit a series of models evaluating these interactions. Interpret and compare the models. In interpreting the estimates, it may help to generate counterfactual preductions contrasting hurricanes with masculine and feminine names. Are the effect sizes plausible?


```R
# first, let's normalize the data
d$f <- standardize(d$femininity)
d$damage <- standardize(d$damage_norm)
d$min_p <- standardize(d$min_pressure)
```


```R
ggplot(d, aes(f, deaths)) +
    geom_point()
```


    
![png](sr_chapter_12_images/output_96_0.png)
    



```R
# need to re-figure out the priors

# figuring out lambda - find the mean number of deaths
mean_deaths <- mean(d$deaths)
log_mean_deaths <- log(mean_deaths)
sd_deaths <- sd(d$deaths)

ALPHA_PRIOR_MEAN <- log_mean_deaths
ALPHA_PRIOR_SD <- 1.2

NUM_SAMPLES <- 1e2
alpha <- rnorm(NUM_SAMPLES, ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD)
lambda <- exp(alpha)
p <- ggplot(d) +
    geom_point(mapping = aes(f, deaths))
femininities <- seq(-2, 2, length.out = 25)
for (i in 1:NUM_SAMPLES) {
    predicted_deaths <- rpois(1, lambda[i])
    p <- p + geom_line(data = data.frame(femininity = femininities, deaths = predicted_deaths), mapping = aes(femininity, deaths), alpha = 0.2)
}
print(p)
```


    
![png](sr_chapter_12_images/output_97_0.png)
    


This looks - decent. Now let's do it for the coefficient!


```R
p <- ggplot(d, aes(f, deaths)) +
    geom_point()

BETA_F_PRIOR_MEAN <- 0
BETA_F_PRIOR_SD <- 0.5

NUM_SAMPLES <- 1e2
alpha <- rnorm(NUM_SAMPLES, ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD)
beta_f <- rnorm(NUM_SAMPLES, BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD)
femininities <- seq(-2, 2, length.out = 25)
for (i in 1:NUM_SAMPLES) {
    # just go with the mean for now
    log_predicted_deaths <- alpha[i] + beta_f[i] * femininities
    predicted_deaths <- exp(log_predicted_deaths)
    p <- p + geom_line(data = data.frame(f = femininities, deaths = predicted_deaths), alpha = 0.2)
}
print(p)
```


    
![png](sr_chapter_12_images/output_99_0.png)
    


This looks reasonable to me! Now we need to grab the priors for the interaction with the pressure. For each, we'll normalize and then use a standard normal as the prior.


```R
interaction_data <- list(
    deaths = d$deaths,
    f = d$f,
    min_p = d$min_p,
    damage = d$damage,
    ALPHA_PRIOR_MEAN = ALPHA_PRIOR_MEAN,
    ALPHA_PRIOR_SD = ALPHA_PRIOR_SD,
    BETA_F_PRIOR_MEAN = BETA_F_PRIOR_MEAN,
    BETA_F_PRIOR_SD = BETA_F_PRIOR_SD
)

m12h3.interaction.damage <- ulam(
    alist(
        deaths ~ dgampois(lambda, phi),
        log(lambda) <- alpha + bf * f + bd * damage + bfd * f * damage,
        alpha ~ dnorm(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD),
        c(bf, bd, bfd) ~ dnorm(BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD),
        phi ~ dexp(1)
    ),
    data = interaction_data,
    log_lik = TRUE
)

m12h3.interaction.pressure <- ulam(
    alist(
        deaths ~ dgampois(lambda, phi),
        log(lambda) <- alpha + bf * f + bp * min_p + bfp * f * min_p,
        alpha ~ dnorm(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD),
        c(bf, bp, bfp) ~ dnorm(BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD),
        phi ~ dexp(1)
    ),
    data = interaction_data,
    log_lik = TRUE
)

m12h3.interaction.both <- ulam(
    alist(
        deaths ~ dgampois(lambda, phi),
        log(lambda) <- alpha + bf * f + bp * min_p + bd * damage + bfp * f * min_p + bfd * f * damage + bfdp * f * damage * min_p,
        alpha ~ dnorm(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD),
        c(bf, bp, bd, bfp, bfd, bfdp) ~ dnorm(BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD),
        phi ~ dexp(1)
    ),
    data = interaction_data,
    log_lik = TRUE
)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    
    Chain 1 finished in 0.3 seconds.

```R
precis(m12h3.interaction.damage)
precis(m12h3.interaction.pressure)
precis(m12h3.interaction.both)
```


<table class="dataframe">
<caption>A precis: 5 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>alpha</th><td>2.61433378</td><td>0.13837266</td><td> 2.38989150</td><td>2.8390660</td><td>0.9996039</td><td>581.0906</td></tr>
	<tr><th scope=row>bfd</th><td>0.25830503</td><td>0.18682629</td><td>-0.02884759</td><td>0.5438960</td><td>0.9996826</td><td>458.9965</td></tr>
	<tr><th scope=row>bd</th><td>1.08969349</td><td>0.19421609</td><td> 0.77856073</td><td>1.4249276</td><td>1.0006224</td><td>521.7022</td></tr>
	<tr><th scope=row>bf</th><td>0.08778099</td><td>0.12790136</td><td>-0.13406371</td><td>0.2814921</td><td>0.9983161</td><td>570.0617</td></tr>
	<tr><th scope=row>phi</th><td>0.67874777</td><td>0.09880588</td><td> 0.52795211</td><td>0.8570944</td><td>1.0026591</td><td>587.4929</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A precis: 5 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>alpha</th><td> 2.7911227</td><td>0.14717601</td><td> 2.56972995</td><td> 3.0428996</td><td>1.0162246</td><td>441.7437</td></tr>
	<tr><th scope=row>bfp</th><td> 0.2818387</td><td>0.14899581</td><td> 0.04163392</td><td> 0.5178375</td><td>1.0067644</td><td>338.1521</td></tr>
	<tr><th scope=row>bp</th><td>-0.6386338</td><td>0.13936584</td><td>-0.85106235</td><td>-0.4119867</td><td>1.0067224</td><td>366.8670</td></tr>
	<tr><th scope=row>bf</th><td> 0.2710867</td><td>0.14174754</td><td> 0.04257546</td><td> 0.4922371</td><td>1.0017188</td><td>479.7444</td></tr>
	<tr><th scope=row>phi</th><td> 0.5539437</td><td>0.07724629</td><td> 0.43689565</td><td> 0.6781608</td><td>0.9981852</td><td>510.7184</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A precis: 8 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>alpha</th><td> 2.5082621</td><td>0.1266655</td><td> 2.30908340</td><td> 2.7055851</td><td>1.0003951</td><td>404.8259</td></tr>
	<tr><th scope=row>bfdp</th><td> 0.2247412</td><td>0.1585285</td><td>-0.01511904</td><td> 0.5005359</td><td>0.9992367</td><td>355.0469</td></tr>
	<tr><th scope=row>bfd</th><td> 0.6506117</td><td>0.2228505</td><td> 0.28548739</td><td> 0.9827494</td><td>1.0051615</td><td>277.6539</td></tr>
	<tr><th scope=row>bfp</th><td> 0.3165563</td><td>0.1561880</td><td> 0.06606444</td><td> 0.5678708</td><td>0.9991921</td><td>343.5489</td></tr>
	<tr><th scope=row>bd</th><td> 0.7502954</td><td>0.1921671</td><td> 0.44715346</td><td> 1.0737409</td><td>0.9984966</td><td>322.7159</td></tr>
	<tr><th scope=row>bp</th><td>-0.5740349</td><td>0.1553959</td><td>-0.80851742</td><td>-0.3199118</td><td>0.9996471</td><td>336.2870</td></tr>
	<tr><th scope=row>bf</th><td> 0.1707738</td><td>0.1229147</td><td>-0.02790241</td><td> 0.3579449</td><td>1.0007921</td><td>571.0509</td></tr>
	<tr><th scope=row>phi</th><td> 0.7593086</td><td>0.1183038</td><td> 0.58668506</td><td> 0.9539645</td><td>1.0111270</td><td>374.7947</td></tr>
</tbody>
</table>




```R
par(bg = 'white')
plot(precis(m12h3.interaction.damage))
plot(precis(m12h3.interaction.pressure))
plot(precis(m12h3.interaction.both))
```


    
![png](sr_chapter_12_images/output_103_0.png)
    



    
![png](sr_chapter_12_images/output_103_1.png)
    



    
![png](sr_chapter_12_images/output_103_2.png)
    


So when we control for only damage (and the interaction with femininity), it looks like just the damage coefficient is relevant. 

If we control for pressure, everything is relevant.

If we control for both then once again femininity becomes less important and it's the other factors (damage and pressure) that are influential.


```R
compare(m12h3.interaction.both, m12h3.interaction.damage, m12h3.interaction.pressure, func = PSIS)
```

    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    
    Some Pareto k values are very high (>1). Set pointwise=TRUE to inspect individual points.
    



<table class="dataframe">
<caption>A compareIC: 3 × 6</caption>
<thead>
	<tr><th></th><th scope=col>PSIS</th><th scope=col>SE</th><th scope=col>dPSIS</th><th scope=col>dSE</th><th scope=col>pPSIS</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m12h3.interaction.both</th><td>665.2222</td><td>35.51491</td><td> 0.000000</td><td>       NA</td><td>10.740911</td><td>9.306444e-01</td></tr>
	<tr><th scope=row>m12h3.interaction.damage</th><td>670.4155</td><td>32.77044</td><td> 5.193271</td><td> 8.623984</td><td> 6.097415</td><td>6.935525e-02</td></tr>
	<tr><th scope=row>m12h3.interaction.pressure</th><td>694.6520</td><td>38.26552</td><td>29.429744</td><td>13.285127</td><td> 8.393696</td><td>3.786135e-07</td></tr>
</tbody>
</table>



From this is looks like the model including both interactions is the best, but not by much.

Now let's see what the model (both interactions) predicts when we change the femininity.


```R
NUM_SAMPLES <- 100
damage_seq <- seq(-2, 2, length.out = NUM_SAMPLES)
pressure_seq <- seq(-2, 2, length.out = NUM_SAMPLES)

male_storm_data <- list(
    damage = rep(0, NUM_SAMPLES),
    min_p = pressure_seq,
    f = rep(-1, NUM_SAMPLES)
)
male_storm_prediction <- link(m12h3.interaction.both, data = male_storm_data)
mean_male_death <- apply(male_storm_prediction, 2, mean)
bounds_male_death <- apply(male_storm_prediction, 2, PI)

female_storm_data <- list(
    damage = rep(0, NUM_SAMPLES),
    min_p = pressure_seq,
    f = rep(1, NUM_SAMPLES)
)
female_storm_prediction <- link(m12h3.interaction.both, data = female_storm_data)
mean_female_death <- apply(female_storm_prediction, 2, mean)
bounds_female_death <- apply(female_storm_prediction, 2, PI)
```


```R
ggplot(d, aes(min_p)) +
    geom_point(mapping = aes(y = deaths)) +
    # male storms
    geom_line(data = data.frame(min_p = pressure_seq, deaths = mean_male_death), mapping = aes(min_p, deaths, colour = "M")) +
    geom_ribbon(data = data.frame(min_p = pressure_seq, death_min = bounds_male_death[1, ], death_max = bounds_male_death[2, ]), mapping = aes(x = min_p, ymin = death_min, ymax = death_max, fill = "M"), alpha = 0.2) +
    # female storms
    geom_line(data = data.frame(min_p = pressure_seq, deaths = mean_female_death), mapping = aes(min_p, deaths, colour = "F")) +
    geom_ribbon(data = data.frame(min_p = pressure_seq, death_min = bounds_female_death[1, ], death_max = bounds_female_death[2, ]), mapping = aes(x = min_p, ymin = death_min, ymax = death_max, fill = "F"), alpha = 0.2)
```


    
![png](sr_chapter_12_images/output_108_0.png)
    


(NB I only did the predictions over pressure mostly due to laziness).

So this prediction tells us a slightly different story: at low pressures we expect male storms to be deadlier but at high pressures (= not as severe) we expect the female ones to be deadler (albeit not by much). Interesting!

#### 6.1.11 12H4 In the original hurricanes paper, storm damage (`damage_norm`) was used directly. This assumption implies that mortality encreases exponentially with a linear increase in storm strength, because a Poisson regression uses a log link. So, it's worth exploring an alternative hypothesis: that the logarithm of storm strength is what matters. Explore this by using the logarithm of `damage_norm` as a predictor. Using the best model structure from the previous problem, compare a model that uses `log(damage_norm)` to a model that uses `damage_norm` directly. Compare their PSIS / WAIC values as well as their implied predictions. What do you conclude?


```R
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 12</caption>
<thead>
	<tr><th></th><th scope=col>name</th><th scope=col>year</th><th scope=col>deaths</th><th scope=col>category</th><th scope=col>min_pressure</th><th scope=col>damage_norm</th><th scope=col>female</th><th scope=col>femininity</th><th scope=col>F</th><th scope=col>f</th><th scope=col>damage</th><th scope=col>min_p</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Easy    </td><td>1950</td><td> 2</td><td>3</td><td>960</td><td> 1590</td><td>1</td><td>6.77778</td><td>-0.0009347453</td><td>-0.0009347453</td><td>-0.4391329</td><td>-0.2576952</td></tr>
	<tr><th scope=row>2</th><td>King    </td><td>1950</td><td> 4</td><td>3</td><td>955</td><td> 5350</td><td>0</td><td>1.38889</td><td>-1.6707580424</td><td>-1.6707580424</td><td>-0.1484282</td><td>-0.5199513</td></tr>
	<tr><th scope=row>3</th><td>Able    </td><td>1952</td><td> 3</td><td>1</td><td>985</td><td>  150</td><td>0</td><td>3.83333</td><td>-0.9133139565</td><td>-0.9133139565</td><td>-0.5504666</td><td> 1.0535855</td></tr>
	<tr><th scope=row>4</th><td>Barbara </td><td>1953</td><td> 1</td><td>1</td><td>987</td><td>   58</td><td>1</td><td>9.83333</td><td> 0.9458703621</td><td> 0.9458703621</td><td>-0.5575796</td><td> 1.1584879</td></tr>
	<tr><th scope=row>5</th><td>Florence</td><td>1953</td><td> 0</td><td>1</td><td>985</td><td>   15</td><td>1</td><td>8.33333</td><td> 0.4810742824</td><td> 0.4810742824</td><td>-0.5609041</td><td> 1.0535855</td></tr>
	<tr><th scope=row>6</th><td>Carol   </td><td>1954</td><td>60</td><td>3</td><td>960</td><td>19321</td><td>1</td><td>8.11111</td><td> 0.4122162926</td><td> 0.4122162926</td><td> 0.9317409</td><td>-0.2576952</td></tr>
</tbody>
</table>




```R
d$l_damage_norm <- log(d$damage_norm)
d$n_l_damage_norm <- standardize(d$l_damage_norm)
summary(d$n_l_damage_norm)
d$n_l_damage_norm
```


        Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    -3.01162 -0.70257  0.09833  0.00000  0.76946  1.70051 



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0.082783428866266</li><li>0.592129460097411</li><li>-0.90825723005481</li><li>-1.30712949676211</li><li>-1.87483773300078</li><li>1.13116803321134</li><li>0.380302125284886</li><li>1.22672533778448</li><li>0.185336570128778</li><li>1.01727889373475</li><li>0.654027252914162</li><li>0.0693707678076311</li><li>-0.370546079073064</li><li>-0.466165349402715</li><li>-0.394540071114927</li><li>1.55690140853453</li><li>-1.51915854679829</li><li>1.04804178716416</li><li>-0.617287505444095</li><li>0.670621531285685</li><li>1.05876239704613</li><li>0.315809212874464</li><li>-0.205554433965096</li><li>1.14566712270542</li><li>-0.243992563457255</li><li>-1.0826825435984</li><li>0.56873505586541</li><li>-0.205554433965096</li><li>1.20506589084879</li><li>0.697102958999771</li><li>-0.617287505444095</li><li>-0.402852829461983</li><li>-0.787493883186525</li><li>1.15459675196252</li><li>-0.0424033580765478</li><li>0.65334964085622</li><li>-0.347849709637109</li><li>-1.25288892134083</li><li>-1.22818882218758</li><li>0.305064699262193</li><li>0.957339448979859</li><li>0.205522217397745</li><li>0.871161491760967</li><li>-0.486158679371905</li><li>-0.968328185017147</li><li>-0.881165212300353</li><li>0.488533778004953</li><li>0.352082247489812</li><li>0.540424515981778</li><li>0.00146904308480593</li><li>-2.25947878672532</li><li>-1.30712949676211</li><li>-3.01162461368918</li><li>-2.42968516446775</li><li>-0.631518718426856</li><li>1.14608669353369</li><li>-0.728824534129797</li><li>0.428153706254904</li><li>1.65147012554195</li><li>-1.09559988828247</li><li>0.0983326353296827</li><li>0.736723301214354</li><li>-0.261608319241605</li><li>0.774451930479087</li><li>-0.787493883186525</li><li>0.0983326353296827</li><li>-0.881165212300353</li><li>0.456186855303573</li><li>-1.10443770058008</li><li>0.76779267243542</li><li>0.0382616803673453</li><li>-0.014866892870573</li><li>-0.693822554072697</li><li>0.562045183942108</li><li>-2.33601383535392</li><li>1.15623732299543</li><li>0.952379400189184</li><li>-0.855716173468071</li><li>1.11497760609341</li><li>0.86342150442053</li><li>-0.552578043852319</li><li>0.297218104848379</li><li>-1.11805339940748</li><li>0.882706682969623</li><li>1.25515623191866</li><li>-1.3611205740609</li><li>-0.0680747972206514</li><li>0.506232067573115</li><li>1.15336210231147</li><li>0.711517442266493</li><li>1.22220217133402</li><li>1.70051455417239</li></ol>




```R
interaction_data[['log_damage']] <- d$n_l_damage_norm

m12h3.interaction.both.log_norm_damage <- ulam(
    alist(
        deaths ~ dgampois(lambda, phi),
        log(lambda) <- alpha + bf * f + bp * min_p + bd * log_damage + bfp * f * min_p + bfd * f * log_damage + bfdp * f * log_damage * min_p,
        alpha ~ dnorm(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD),
        c(bf, bp, bd, bfp, bfd, bfdp) ~ dnorm(BETA_F_PRIOR_MEAN, BETA_F_PRIOR_SD),
        phi ~ dexp(1)
    ),
    data = interaction_data,
    log_lik = TRUE
)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    Chain 1 finished in 0.3 seconds.

```R
precis(m12h3.interaction.both.log_norm_damage)
par(bg = 'white')
plot(precis(m12h3.interaction.both.log_norm_damage))
```

<table class="dataframe">
<caption>A precis: 8 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>alpha</th><td> 2.29386676</td><td>0.1202504</td><td> 2.1096958</td><td>2.49318610</td><td>1.0044328</td><td>380.7294</td></tr>
	<tr><th scope=row>bfdp</th><td>-0.02359698</td><td>0.1317901</td><td>-0.2271794</td><td>0.20111247</td><td>1.0026978</td><td>302.7689</td></tr>
	<tr><th scope=row>bfd</th><td> 0.20444878</td><td>0.1977236</td><td>-0.1097949</td><td>0.52791755</td><td>1.0204603</td><td>289.2747</td></tr>
	<tr><th scope=row>bfp</th><td> 0.05473029</td><td>0.1868201</td><td>-0.2398280</td><td>0.35589775</td><td>1.0078622</td><td>283.5645</td></tr>
	<tr><th scope=row>bd</th><td> 1.20157472</td><td>0.1665740</td><td> 0.9499617</td><td>1.48735000</td><td>1.0056547</td><td>310.1423</td></tr>
	<tr><th scope=row>bp</th><td>-0.14723247</td><td>0.1557312</td><td>-0.3852782</td><td>0.08627849</td><td>1.0090310</td><td>326.4621</td></tr>
	<tr><th scope=row>bf</th><td> 0.01710455</td><td>0.1455624</td><td>-0.2067255</td><td>0.26036785</td><td>0.9985165</td><td>273.7513</td></tr>
	<tr><th scope=row>phi</th><td> 1.00267347</td><td>0.1523869</td><td> 0.7726367</td><td>1.25177535</td><td>0.9982541</td><td>428.8417</td></tr>
</tbody>
</table>




    
![png](sr_chapter_12_images/output_114_1.png)
    


Very interesting! So now that we're using the log of the damage, we're pretty sure that only the damage is relevant.


```R
compare(m12h3.interaction.both, m12h3.interaction.both.log_norm_damage)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m12h3.interaction.both.log_norm_damage</th><td>637.8814</td><td>32.08560</td><td> 0.00000</td><td>      NA</td><td> 8.680689</td><td>9.999984e-01</td></tr>
	<tr><th scope=row>m12h3.interaction.both</th><td>664.5345</td><td>35.15605</td><td>26.65316</td><td>11.20777</td><td>10.397064</td><td>1.630566e-06</td></tr>
</tbody>
</table>



The model that uses the log damage also has a smaller WAIC value (although less than one standard error) - it is doing a slightly better job explaining the data.

#### 6.1.12 12H5 One hypothesis from developmental psychology, usually attributed to Carol Gillian, proposes that women and men have different average tendencies in moral reasoning. Like most hypotheses in social psychology, it is descriptive, not causal. The notion is that women are more concerned with care (avoiding harm) while men are more concerned with justice and rights. Evaluate this hypothesis using the `Trolley` data, supposing that `contact` provides a proxy for physical harm. Are women more or less bothered by contact than are men, in these data? Figure out the model(s) that is needed to address the question.

We'll use the same model, but now give different values for the coefficients depending on male / female.


```R
data(Trolley)
d <- Trolley
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 12</caption>
<thead>
	<tr><th></th><th scope=col>case</th><th scope=col>response</th><th scope=col>order</th><th scope=col>id</th><th scope=col>age</th><th scope=col>male</th><th scope=col>edu</th><th scope=col>action</th><th scope=col>intention</th><th scope=col>contact</th><th scope=col>story</th><th scope=col>action2</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>cfaqu</td><td>4</td><td> 2</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>0</td><td>1</td><td>aqu</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>cfbur</td><td>3</td><td>31</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>0</td><td>1</td><td>bur</td><td>1</td></tr>
	<tr><th scope=row>3</th><td>cfrub</td><td>4</td><td>16</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>0</td><td>1</td><td>rub</td><td>1</td></tr>
	<tr><th scope=row>4</th><td>cibox</td><td>3</td><td>32</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>1</td><td>1</td><td>box</td><td>1</td></tr>
	<tr><th scope=row>5</th><td>cibur</td><td>3</td><td> 4</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>1</td><td>1</td><td>bur</td><td>1</td></tr>
	<tr><th scope=row>6</th><td>cispe</td><td>3</td><td> 9</td><td>96;434</td><td>14</td><td>0</td><td>Middle School</td><td>0</td><td>1</td><td>1</td><td>spe</td><td>1</td></tr>
</tbody>
</table>




```R

dat <- list(
    R = d$response,
    A = d$action,
    I = d$intention,
    C = d$contact,
    Gid = ifelse(d$male == 0, 1L, 2L) # index - 1 for male, 2 for female
)
lapply(dat, head)
dat$Gid
```


<dl>
	<dt>$R</dt>
		<dd><style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>4</li><li>3</li><li>4</li><li>3</li><li>3</li><li>3</li></ol>
</dd>
	<dt>$A</dt>
		<dd><style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0</li><li>0</li><li>0</li><li>0</li><li>0</li><li>0</li></ol>
</dd>
	<dt>$I</dt>
		<dd><style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0</li><li>0</li><li>0</li><li>1</li><li>1</li><li>1</li></ol>
</dd>
	<dt>$C</dt>
		<dd><style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li></ol>
</dd>
	<dt>$Gid</dt>
		<dd><style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li></ol>
</dd>
</dl>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>⋯</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li></ol>




```R
m12h5 <- ulam(
    alist(
        R ~ dordlogit(phi, cutpoints),
        phi <- bA[Gid] * A + bC[Gid] * C + BI * I,
        BI <- bI + bIA[Gid] * A + bIC[Gid] * C,
        bI ~ dnorm(0, 0.5),
        bA[Gid] ~ dnorm(0, 0.5),
        bC[Gid] ~ dnorm(0, 0.5),
        bIC[Gid] ~ dnorm(0, 0.5),
        bIA[Gid] ~ dnorm(0, 0.5),
        cutpoints ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 4,
    cores = 4
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 192.3 seconds.
    Total execution time: 201.2 seconds.

<table class="dataframe">
<caption>A precis: 15 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>bI</th><td>-0.3065778</td><td>0.05640764</td><td>-0.3942669</td><td>-0.215918680</td><td>1.0028049</td><td> 764.0882</td></tr>
	<tr><th scope=row>bA[1]</th><td>-0.7562650</td><td>0.06577073</td><td>-0.8584842</td><td>-0.653486665</td><td>1.0016383</td><td> 912.3648</td></tr>
	<tr><th scope=row>bA[2]</th><td>-0.2452401</td><td>0.06056232</td><td>-0.3408817</td><td>-0.150372465</td><td>1.0039680</td><td> 837.0125</td></tr>
	<tr><th scope=row>bC[1]</th><td>-0.6198487</td><td>0.08900053</td><td>-0.7604721</td><td>-0.478074845</td><td>0.9995898</td><td>1317.2669</td></tr>
	<tr><th scope=row>bC[2]</th><td>-0.1346086</td><td>0.08353697</td><td>-0.2656810</td><td> 0.000352656</td><td>1.0045355</td><td>1005.7429</td></tr>
	<tr><th scope=row>bIC[1]</th><td>-1.1015761</td><td>0.12430514</td><td>-1.2939243</td><td>-0.900461960</td><td>1.0022651</td><td>1298.7577</td></tr>
	<tr><th scope=row>bIC[2]</th><td>-1.2984257</td><td>0.11776713</td><td>-1.4898769</td><td>-1.120074350</td><td>1.0028564</td><td>1088.3760</td></tr>
	<tr><th scope=row>bIA[1]</th><td>-0.3712081</td><td>0.09764806</td><td>-0.5269281</td><td>-0.216332850</td><td>1.0019713</td><td> 947.3549</td></tr>
	<tr><th scope=row>bIA[2]</th><td>-0.4548537</td><td>0.09250325</td><td>-0.6047378</td><td>-0.310297345</td><td>1.0067107</td><td> 822.2111</td></tr>
	<tr><th scope=row>cutpoints[1]</th><td>-2.6512861</td><td>0.05093094</td><td>-2.7313171</td><td>-2.568570100</td><td>1.0017053</td><td> 820.9481</td></tr>
	<tr><th scope=row>cutpoints[2]</th><td>-1.9575335</td><td>0.04681867</td><td>-2.0307237</td><td>-1.882871750</td><td>1.0021271</td><td> 718.0575</td></tr>
	<tr><th scope=row>cutpoints[3]</th><td>-1.3598617</td><td>0.04510828</td><td>-1.4317011</td><td>-1.286116250</td><td>1.0022255</td><td> 642.4998</td></tr>
	<tr><th scope=row>cutpoints[4]</th><td>-0.3146359</td><td>0.04357853</td><td>-0.3835308</td><td>-0.245026355</td><td>1.0027128</td><td> 664.7696</td></tr>
	<tr><th scope=row>cutpoints[5]</th><td> 0.3642290</td><td>0.04375144</td><td> 0.2949406</td><td> 0.434527900</td><td>1.0028631</td><td> 836.8323</td></tr>
	<tr><th scope=row>cutpoints[6]</th><td> 1.2774009</td><td>0.04603521</td><td> 1.2065089</td><td> 1.352401100</td><td>1.0014592</td><td> 950.6563</td></tr>
</tbody>
</table>

    14 vector or matrix parameters hidden. Use depth=2 to show them.

    
![png](sr_chapter_12_images/output_122_11.png)
    

```R
prec <- precis(m12h5, depth = 2)

print(prec)
par(bg = 'white')
plot(prec)
```

                       mean         sd       5.5%        94.5%      rhat  ess_bulk
    bI           -0.3065778 0.05640764 -0.3942669 -0.215918680 1.0028049  764.0882
    bA[1]        -0.7562650 0.06577073 -0.8584842 -0.653486665 1.0016383  912.3648
    bA[2]        -0.2452401 0.06056232 -0.3408817 -0.150372465 1.0039680  837.0125
    bC[1]        -0.6198487 0.08900053 -0.7604721 -0.478074845 0.9995898 1317.2669
    bC[2]        -0.1346086 0.08353697 -0.2656810  0.000352656 1.0045355 1005.7429
    bIC[1]       -1.1015761 0.12430514 -1.2939243 -0.900461960 1.0022651 1298.7577
    bIC[2]       -1.2984257 0.11776713 -1.4898769 -1.120074350 1.0028564 1088.3760
    bIA[1]       -0.3712081 0.09764806 -0.5269281 -0.216332850 1.0019713  947.3549
    bIA[2]       -0.4548537 0.09250325 -0.6047378 -0.310297345 1.0067107  822.2111
    cutpoints[1] -2.6512861 0.05093094 -2.7313171 -2.568570100 1.0017053  820.9481
    cutpoints[2] -1.9575335 0.04681867 -2.0307237 -1.882871750 1.0021271  718.0575
    cutpoints[3] -1.3598617 0.04510828 -1.4317011 -1.286116250 1.0022255  642.4998
    cutpoints[4] -0.3146359 0.04357853 -0.3835308 -0.245026355 1.0027128  664.7696
    cutpoints[5]  0.3642290 0.04375144  0.2949406  0.434527900 1.0028631  836.8323
    cutpoints[6]  1.2774009 0.04603521  1.2065089  1.352401100 1.0014592  950.6563



    
![png](sr_chapter_12_images/output_123_1.png)
    


This data / model (roughly) supports the hypothesis that females are more concerned about harm (proxy: the 'contact' variable) than men. Looking at `bC`, the male parameter is less than the female one.

#### 6.1.13 12H6 The data in `data(Fish)` are records of visits to a national park. The question of interest is how many fish an average visitor takes per hour when fishing. The problem is that not everyone tried to fish, so the `fish_caught` numbers are zero-inflated. As with the monks example in the chapter, there is a process that determines who is fishing (working) and another process that determines fish per hour (manuscripts per day), conditional on fishing (working). We want to model both. Otherwise we'll end up with an underestimate of rate of fish extraction from the park. You will model these data using zero-inflated Poisson GLMs. Predict `fish_caught` as a function of any of the other cariables you think are relevant. One thing you must do, however, is to use a proper Poisson offset / exposure in the Poisson portion of the zero-inflated model. Then use the `hours` variable to construct the offset. This will adjust the model for the differing amount of time individuals spent in the park.


```R
data(Fish)
d <- Fish
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 6</caption>
<thead>
	<tr><th></th><th scope=col>fish_caught</th><th scope=col>livebait</th><th scope=col>camper</th><th scope=col>persons</th><th scope=col>child</th><th scope=col>hours</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>21.124</td></tr>
	<tr><th scope=row>2</th><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td> 5.732</td></tr>
	<tr><th scope=row>3</th><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td><td> 1.323</td></tr>
	<tr><th scope=row>4</th><td>0</td><td>1</td><td>1</td><td>2</td><td>1</td><td> 0.548</td></tr>
	<tr><th scope=row>5</th><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td> 1.695</td></tr>
	<tr><th scope=row>6</th><td>0</td><td>1</td><td>1</td><td>4</td><td>2</td><td> 0.493</td></tr>
</tbody>
</table>




```R
summary(d)
```


      fish_caught         livebait         camper         persons     
     Min.   :  0.000   Min.   :0.000   Min.   :0.000   Min.   :1.000  
     1st Qu.:  0.000   1st Qu.:1.000   1st Qu.:0.000   1st Qu.:2.000  
     Median :  0.000   Median :1.000   Median :1.000   Median :2.000  
     Mean   :  3.296   Mean   :0.864   Mean   :0.588   Mean   :2.528  
     3rd Qu.:  2.000   3rd Qu.:1.000   3rd Qu.:1.000   3rd Qu.:4.000  
     Max.   :149.000   Max.   :1.000   Max.   :1.000   Max.   :4.000  
         child           hours        
     Min.   :0.000   Min.   : 0.0040  
     1st Qu.:0.000   1st Qu.: 0.2865  
     Median :0.000   Median : 1.8315  
     Mean   :0.684   Mean   : 5.5260  
     3rd Qu.:1.000   3rd Qu.: 7.3427  
     Max.   :3.000   Max.   :71.0360  



```R
?Fish
```

    Fish                package:rethinking                 R Documentation
    
    _F_i_s_h_i_n_g _d_a_t_a
    
    _D_e_s_c_r_i_p_t_i_o_n:
    
         Fishing data from visitors to a national park.
    
    _U_s_a_g_e:
    
         data(Fish)
         
    _F_o_r_m_a_t:
    
           1. fish_caught : Number of fish caught during visit
    
           2. livebait : Whether or not group used livebait to fish
    
           3. camper : Whether or not group had a camper
    
           4. persons : Number of adults in group
    
           5. child : Number of children in group
    
           6. hours : Number of hours group spent in park



```R
ggplot(d, aes(fish_caught)) +
    geom_histogram()
```

    [1m[22m`stat_bin()` using `bins = 30`. Pick better value with `binwidth`.



    
![png](sr_chapter_12_images/output_129_1.png)
    



```R
ggplot(d, aes(hours, fish_caught)) +
    geom_point()
```


    
![png](sr_chapter_12_images/output_130_0.png)
    


Let's start with a super simple model: zero inflated, some probability of fishing and then some prior for the rate lambda.


```R
data <- list(
    f = d$fish_caught,
    log_hours = log(d$hours)
)
m12h6.1 <- ulam(
    alist(
        f ~ dzipois(p, lambda),
        logit(p) ~ dnorm(0, 1),
        log(lambda) <- log_hours + a,
        a ~ dnorm(0, 1)
    ),
    data = data,
    log_lik = TRUE
)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    Chain 1 finished in 0.3 seconds.

```R
prec <- precis(m12h6.1)
print(prec)
par(bg = 'white')
plot(prec)
```

            mean         sd       5.5%       94.5%      rhat ess_bulk
    p  0.3233473 0.03992419  0.2612199  0.38644382 1.0037633 186.6498
    a -0.1458901 0.03796932 -0.2104045 -0.08749367 0.9982549 232.9276



    
![png](sr_chapter_12_images/output_133_1.png)
    



```R
hours_sim <- seq(1, 50, length.out = 100)
preds <- link(m12h6.1, data = list(log_hours=(log(hours_sim))))
preds
```


<table class="dataframe">
<caption>A matrix: 500 × 100 of type dbl</caption>
<tbody>
	<tr><td>0.8492370</td><td>1.269566</td><td>1.689896</td><td>2.110225</td><td>2.530555</td><td>2.950884</td><td>3.371213</td><td>3.791543</td><td>4.211872</td><td>4.632202</td><td>⋯</td><td>38.67888</td><td>39.09921</td><td>39.51954</td><td>39.93987</td><td>40.36020</td><td>40.78053</td><td>41.20086</td><td>41.62119</td><td>42.04152</td><td>42.46185</td></tr>
	<tr><td>0.8311234</td><td>1.242488</td><td>1.653852</td><td>2.065216</td><td>2.476580</td><td>2.887944</td><td>3.299308</td><td>3.710672</td><td>4.122036</td><td>4.533400</td><td>⋯</td><td>37.85389</td><td>38.26526</td><td>38.67662</td><td>39.08799</td><td>39.49935</td><td>39.91071</td><td>40.32208</td><td>40.73344</td><td>41.14481</td><td>41.55617</td></tr>
</tbody>
</table>




```R
means <- apply(preds, 2, mean)
bounds <- apply(preds, 2, PI)

plot_df <- data.frame(hours = hours_sim, mean = means, lower = bounds[1, ], upper = bounds[2, ])
ggplot(plot_df, aes(hours)) +
    geom_point(data = d, aes(hours, fish_caught)) +
    geom_line(mapping = aes(y = mean)) +
    geom_ribbon(data = plot_df, aes(hours, ymin = lower, ymax = upper), alpha = 0.2)
```


    
![png](sr_chapter_12_images/output_135_0.png)
    


This is pretty bad. Let's add in some predictors.

- `live_bait` should increase the number of fish caught
- `camper` should affect the probability of going fishing
- `persons` should affect the number caught
- `child` and adults should have a different 'rate' of catching fish


```R
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 6</caption>
<thead>
	<tr><th></th><th scope=col>fish_caught</th><th scope=col>livebait</th><th scope=col>camper</th><th scope=col>persons</th><th scope=col>child</th><th scope=col>hours</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>21.124</td></tr>
	<tr><th scope=row>2</th><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td> 5.732</td></tr>
	<tr><th scope=row>3</th><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td><td> 1.323</td></tr>
	<tr><th scope=row>4</th><td>0</td><td>1</td><td>1</td><td>2</td><td>1</td><td> 0.548</td></tr>
	<tr><th scope=row>5</th><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td> 1.695</td></tr>
	<tr><th scope=row>6</th><td>0</td><td>1</td><td>1</td><td>4</td><td>2</td><td> 0.493</td></tr>
</tbody>
</table>




```R
data <- list(
    f = d$fish_caught,
    log_hours = log(d$hours),
    livebait_index = d$livebait + 1,
    camper_index = d$camper + 1,
    adults = d$persons - d$child,
    children = d$child
)

m12h6.2 <- ulam(
    alist(
        f ~ dzipois(p, lambda),
        logit(p) <- bc[camper_index],
        log(lambda) <- log_hours + a[livebait_index] + b_adult * adults + b_child * children,
        bc[camper_index] ~ dnorm(0, 1),
        a[livebait_index] ~ dnorm(0, 1),
        c(b_adult, b_child) ~ dnorm(0, 1)
    ),
    data = data,
    log_lik = TRUE
)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    
    Chain 1 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 1 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 1 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 1 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 1 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 1 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 1 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 1 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 1 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 1 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 1 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 1 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 1 finished in 1.3 seconds.



```R
pre <- precis(m12h6.2, depth = 2)
print(pre)
par(bg = 'white')
plot(pre)
```

                  mean         sd       5.5%      94.5%      rhat ess_bulk
    bc[1]   -1.1755715 0.43875374 -1.9319694 -0.5453518 1.0069789 439.4500
    bc[2]   -1.1881850 0.29246003 -1.6784902 -0.7316058 0.9980242 428.6331
    a[1]    -3.4929129 0.26972767 -3.9200211 -3.0523188 1.0057097 295.1996
    a[2]    -2.1734427 0.14612443 -2.4154674 -1.9339698 1.0005593 272.1121
    b_child  1.1901226 0.09689764  1.0206932  1.3388843 0.9986143 379.3646
    b_adult  0.6540439 0.04169641  0.5887261  0.7182928 0.9998876 269.3255



    
![png](sr_chapter_12_images/output_139_1.png)
    



```R
compare(m12h6.1, m12h6.2)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m12h6.2</th><td>2146.362</td><td>391.5616</td><td>  0.0000</td><td>      NA</td><td>91.57640</td><td>1.000000e+00</td></tr>
	<tr><th scope=row>m12h6.1</th><td>2563.622</td><td>466.4085</td><td>417.2603</td><td>227.3142</td><td>45.19352</td><td>2.472103e-91</td></tr>
</tbody>
</table>



Some interesting results! It looks like being in a camper doesn't actually affect your chances of going fishing, which is a bit surprising. Also surprising is the fact that it looks like children are better at fishing than adults!


```R
child_data <- list(
    log_hours = log(24),
    livebait_index = 1,
    camper_index = 1,
    adults = 0,
    children = 1
)
adult_data <- list(
    log_hours = log(24),
    livebait_index = 1,
    camper_index = 1,
    adults = 1,
    children = 0
)

l_child <- link(m12h6.2, data = child_data)
child_counts <- rbern(length(l_child$p), l_child$p) * rpois(length(l_child$lambda), l_child$lambda)
l_adult <- link(m12h6.2, data = adult_data)
adult_counts <- rbern(length(l_adult$p), l_adult$p) * rpois(length(l_adult$lambda), l_adult$lambda)

plot_df <- rbind(
    data.frame(count = child_counts, type = "Child"),
    data.frame(count = adult_counts, type = "Adult")
)

ggplot(plot_df, aes(count, colour = type, fill = type)) +
    geom_bar(position = 'dodge')
```


    
![png](sr_chapter_12_images/output_142_0.png)
    


#### 6.1.14 12H7 In the trolley data we saw how education level (modeled as an ordered category) is associated with responses. But is this association causal? One plausible confound is that education is also associated with age, through a causal process: prople are older when they finish school than when they begin it. Reconsider the Trolley data in this light. Draw a DAG that representat hypotheical causal relathionships among response, education, and age. Which statistical model or models do you need to evaluate the causal influence of education on responses? Fit these models to the trolley data. What do you conclude about the causal relationships among these three variables?

#### 6.1.15 12H8 Consider one more variable in the trolley data: Gender. Suppose that gender might influence education as well as response directly. Draw the DAG now that includes responses, education, age, and gender. Using only the DAG, is it possible that the inferences from 12H7 above are confounded by gender? If so, define any additional omdels you need to infer the causal influence of education on response. What do you conclude?
