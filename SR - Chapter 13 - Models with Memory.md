# Chapter 13 - Models with Memory

- Most models so far 'forget' everything about previous clusters of data as soon as they move on to the next
- They have 'anterograde amnesia'
- Say you have a robot going to cafes and ordering coffee. They want to estimate the wait times. Say they start with a prior of 5 minutes and an sd of 1 minute. Then they order coffee at one cafe, and it takes 4 minutes. They update their prior and move on to the next cafe. What should the prior at this cafe be? We kind of want to estimate parameters for *each* cafe as well as parameters for the population of cafes as a whole. As the robot goes to the cafes, it should update both the population-level and cafe-level parameters, using the population level as the prior for each new cafe. If the cafes are all different this will not provide much information, but if they are all similar then it provides a lot of information!
- The formal version of this argument leads us to **multilevel models**
- They 'remember' features of each cluster as they learn about all of the clusters
- Tends to improve estimates about each cluster
- Several other benefits:
1. Improved estimates for repeat sampling. When multiple observations arise from the same individual (or whatever), single-level models tend to underfit / overfit the model
1. Improved estimates for imbalances in sampling. When some individuals are sampled more than others, multilevel models automatically cope with this. This provents over-sampled clusters for unfairly dominating inferences.
1. Estimates of variance. If our research questions include variation among individuals or other groups within the data, then multilevel models are a big help because they model variation explicitly.
1. Acoid averaging, retain variation. Frequently, people average some value to construct variables. This can be dangerous because the averaging removes information about the variation. Also, there are several different ways to construct the average. **Multilevel models** allow us to preserve uncertainty and avoid data transformations.

In general, there is a strong argument that multilevel models should be the default over single-level ones.

Of course, there are some downsides.
1. New assumptions. We need to define the distributions from which the characteristics of the cluster arise.
1. Challenges with estimation. Apparently this is harder than with a single-level model?
1. Difficult to understand. Multilevel models make predictions at different levels of the model, and so can be difficult to interpret.

**Multilevel models** are also called **Hierarchical models** or **Mixed effects models**. The type of parameters that appear in the models are sometimes known as **random effects**. There is a lot of vocabulary around this and it is frequently inconsistently used - basically you often need to look at the actual math to figure out what is going on!

## 1 Example: multilevel tadpoles

We'll look at frog mortality.


```R
library(rethinking)
library(ggplot2)
library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

options(repr.plot.width = 17, repr.plot.height = 8)
```

```R
data(reedfrogs)
d <- reedfrogs
str(d)
```

    'data.frame':	48 obs. of  5 variables:
     $ density : int  10 10 10 10 10 10 10 10 10 10 ...
     $ pred    : Factor w/ 2 levels "no","pred": 1 1 1 1 1 1 1 1 2 2 ...
     $ size    : Factor w/ 2 levels "big","small": 1 1 1 1 2 2 2 2 1 1 ...
     $ surv    : int  9 10 7 10 9 9 10 9 4 9 ...
     $ propsurv: num  0.9 1 0.7 1 0.9 0.9 1 0.9 0.4 0.9 ...


We're interested in the number surviving (`surv`) out of the initial population (`density`).

We can think of each row as a 'tank' containing some tadpoles. There are a lot of thinkgs that are unique to each tank, so even when all of our recorded predictor variables are the same there are still differences. These tanks are an example of a 'cluster', and we make multiple observations within each cluster.

If we ignore the clusering and assign the same intercept to each, we ignore important variation in the baseline survival.

If we go the other direction and assign a unique intercept to each tank, then we practice 'anterograde amnesia' - information from tank within the same cluster should really inform our views. Each tank is different, but knowing information about one tank should also give us information about other tanks in the same cluster.

We want a multilevel model, where we will simulatenously estimate both an intercept for *each* tank and a variation *among tanks*. This will be a [[Varying Intercepts Model]]. Varying intercepts are the simplest kind of [[Varying Effect Model]]. For each cluster in the data, we use a unique intercept parameter. This is essentially the same as the categorical variable model from earlier, but now we are also adaptively learning the prior that is common to the intercepts.

Here's the model for predicting tadpole mortality:

$$
\begin{align*}
S_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} \\
\alpha_j &\sim \text{Normal}(0, 1.5) & \text{for $j = 1\dots 48$}
\end{align*}
$$


```R
# tank cluster variable
d$tank <- 1:nrow(d)

dat <- list(
    S = d$surv,
    N = d$density,
    tank = d$tank
)

m13.1 <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank],
        a[tank] ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.6 seconds.

```R
par(bg = 'white')
precis(m13.1, depth = 2)
plot(precis(m13.1, depth = 2))
```


<table class="dataframe">
<caption>A precis: 48 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td> 1.70049950</td><td>0.7497252</td><td> 0.59887018</td><td> 2.975501850</td><td>1.0022289</td><td>4889.511</td></tr>
	<tr><th scope=row>a[2]</th><td> 2.39518204</td><td>0.8876435</td><td> 1.06426105</td><td> 3.930386750</td><td>1.0048374</td><td>4655.833</td></tr>
	<tr><th scope=row>a[3]</th><td> 0.75640364</td><td>0.6306012</td><td>-0.19681222</td><td> 1.788293300</td><td>1.0048841</td><td>4160.963</td></tr>
	<tr><th scope=row>a[4]</th><td> 2.38356397</td><td>0.8684106</td><td> 1.11627085</td><td> 3.822064950</td><td>1.0000318</td><td>5400.087</td></tr>
	<tr><th scope=row>a[5]</th><td> 1.69594537</td><td>0.7583699</td><td> 0.57999923</td><td> 2.998542100</td><td>1.0028963</td><td>4331.713</td></tr>
	<tr><th scope=row>a[6]</th><td> 1.71122167</td><td>0.7363207</td><td> 0.57281489</td><td> 2.955731450</td><td>0.9991404</td><td>4016.962</td></tr>
	<tr><th scope=row>a[7]</th><td> 2.38899473</td><td>0.9123562</td><td> 1.03610515</td><td> 3.948461200</td><td>1.0053820</td><td>4172.170</td></tr>
	<tr><th scope=row>a[8]</th><td> 1.71372195</td><td>0.7340980</td><td> 0.58567738</td><td> 2.985565450</td><td>1.0046840</td><td>3330.060</td></tr>
	<tr><th scope=row>a[9]</th><td>-0.38094600</td><td>0.6181678</td><td>-1.38565050</td><td> 0.603662830</td><td>1.0027876</td><td>3984.459</td></tr>
	<tr><th scope=row>a[10]</th><td> 1.72310683</td><td>0.8126847</td><td> 0.51113849</td><td> 3.093611850</td><td>1.0042392</td><td>3214.386</td></tr>
	<tr><th scope=row>a[11]</th><td> 0.74239817</td><td>0.6145996</td><td>-0.21912968</td><td> 1.729549100</td><td>1.0038181</td><td>4787.070</td></tr>
	<tr><th scope=row>a[12]</th><td> 0.36779522</td><td>0.5881692</td><td>-0.52664893</td><td> 1.297336600</td><td>1.0014061</td><td>5254.104</td></tr>
	<tr><th scope=row>a[13]</th><td> 0.76343334</td><td>0.6263288</td><td>-0.19819923</td><td> 1.796871650</td><td>1.0041913</td><td>5228.821</td></tr>
	<tr><th scope=row>a[14]</th><td> 0.01954581</td><td>0.5841969</td><td>-0.93606455</td><td> 0.939591940</td><td>1.0049121</td><td>5817.342</td></tr>
	<tr><th scope=row>a[15]</th><td> 1.70949390</td><td>0.7547426</td><td> 0.56800733</td><td> 2.994540950</td><td>1.0024950</td><td>4341.449</td></tr>
	<tr><th scope=row>a[16]</th><td> 1.73520113</td><td>0.7487890</td><td> 0.61776586</td><td> 2.992198100</td><td>1.0020416</td><td>4315.823</td></tr>
	<tr><th scope=row>a[17]</th><td> 2.52971530</td><td>0.6377864</td><td> 1.57803810</td><td> 3.596058500</td><td>1.0016941</td><td>4681.037</td></tr>
	<tr><th scope=row>a[18]</th><td> 2.13772481</td><td>0.6087378</td><td> 1.22904295</td><td> 3.141571300</td><td>1.0069333</td><td>4777.709</td></tr>
	<tr><th scope=row>a[19]</th><td> 1.81810045</td><td>0.5604278</td><td> 0.96183403</td><td> 2.753333850</td><td>1.0014860</td><td>5290.065</td></tr>
	<tr><th scope=row>a[20]</th><td> 3.07671203</td><td>0.7825060</td><td> 1.97118425</td><td> 4.440145500</td><td>1.0017148</td><td>4706.840</td></tr>
	<tr><th scope=row>a[21]</th><td> 2.14771465</td><td>0.5951666</td><td> 1.26815465</td><td> 3.165760200</td><td>1.0061698</td><td>4827.711</td></tr>
	<tr><th scope=row>a[22]</th><td> 2.14409686</td><td>0.6256160</td><td> 1.21389430</td><td> 3.145202100</td><td>1.0008013</td><td>6602.060</td></tr>
	<tr><th scope=row>a[23]</th><td> 2.15442374</td><td>0.6014744</td><td> 1.27542770</td><td> 3.173858150</td><td>1.0032253</td><td>3846.463</td></tr>
	<tr><th scope=row>a[24]</th><td> 1.53726624</td><td>0.5253774</td><td> 0.73053575</td><td> 2.437899250</td><td>1.0011781</td><td>4762.780</td></tr>
	<tr><th scope=row>a[25]</th><td>-1.09762863</td><td>0.4358341</td><td>-1.82144575</td><td>-0.411782295</td><td>0.9990513</td><td>4586.819</td></tr>
	<tr><th scope=row>a[26]</th><td> 0.07148796</td><td>0.3930504</td><td>-0.51436716</td><td> 0.675233055</td><td>1.0049674</td><td>4712.190</td></tr>
	<tr><th scope=row>a[27]</th><td>-1.54921715</td><td>0.4900185</td><td>-2.34879410</td><td>-0.786765915</td><td>1.0037255</td><td>5115.843</td></tr>
	<tr><th scope=row>a[28]</th><td>-0.55985854</td><td>0.4134748</td><td>-1.23690195</td><td> 0.085239247</td><td>1.0045258</td><td>5233.299</td></tr>
	<tr><th scope=row>a[29]</th><td> 0.08064254</td><td>0.3963550</td><td>-0.53698910</td><td> 0.721634190</td><td>1.0014990</td><td>5547.496</td></tr>
	<tr><th scope=row>a[30]</th><td> 1.30491775</td><td>0.4581103</td><td> 0.60749076</td><td> 2.078997100</td><td>1.0015329</td><td>6247.562</td></tr>
	<tr><th scope=row>a[31]</th><td>-0.72625624</td><td>0.4044059</td><td>-1.38033420</td><td>-0.097516358</td><td>1.0042292</td><td>5340.167</td></tr>
	<tr><th scope=row>a[32]</th><td>-0.39313426</td><td>0.4086365</td><td>-1.04187610</td><td> 0.250583270</td><td>0.9993788</td><td>5578.839</td></tr>
	<tr><th scope=row>a[33]</th><td> 2.85396841</td><td>0.6585512</td><td> 1.87070620</td><td> 3.913768050</td><td>1.0033343</td><td>3776.145</td></tr>
	<tr><th scope=row>a[34]</th><td> 2.48506977</td><td>0.5990868</td><td> 1.57997330</td><td> 3.472322100</td><td>1.0064588</td><td>4546.533</td></tr>
	<tr><th scope=row>a[35]</th><td> 2.45485111</td><td>0.5878307</td><td> 1.59663395</td><td> 3.462075700</td><td>0.9997085</td><td>5028.903</td></tr>
	<tr><th scope=row>a[36]</th><td> 1.90380406</td><td>0.4782167</td><td> 1.18059355</td><td> 2.698417900</td><td>0.9996165</td><td>5357.651</td></tr>
	<tr><th scope=row>a[37]</th><td> 1.90285691</td><td>0.4705635</td><td> 1.17743465</td><td> 2.676869000</td><td>1.0004010</td><td>3653.147</td></tr>
	<tr><th scope=row>a[38]</th><td> 3.36315848</td><td>0.8060520</td><td> 2.21079635</td><td> 4.800772050</td><td>1.0067639</td><td>5101.424</td></tr>
	<tr><th scope=row>a[39]</th><td> 2.46614807</td><td>0.5643787</td><td> 1.63177055</td><td> 3.389822350</td><td>1.0044997</td><td>4593.130</td></tr>
	<tr><th scope=row>a[40]</th><td> 2.16239378</td><td>0.5206009</td><td> 1.38778770</td><td> 3.045046250</td><td>0.9993820</td><td>3638.898</td></tr>
	<tr><th scope=row>a[41]</th><td>-1.91896826</td><td>0.4920083</td><td>-2.75053900</td><td>-1.152667450</td><td>1.0009009</td><td>4259.936</td></tr>
	<tr><th scope=row>a[42]</th><td>-0.63372891</td><td>0.3477850</td><td>-1.19529870</td><td>-0.098636503</td><td>1.0050435</td><td>5039.642</td></tr>
	<tr><th scope=row>a[43]</th><td>-0.51291731</td><td>0.3322768</td><td>-1.06292195</td><td>-0.008861441</td><td>1.0022415</td><td>6033.443</td></tr>
	<tr><th scope=row>a[44]</th><td>-0.39507784</td><td>0.3340188</td><td>-0.92250193</td><td> 0.129665570</td><td>1.0036692</td><td>5041.826</td></tr>
	<tr><th scope=row>a[45]</th><td> 0.51107729</td><td>0.3459851</td><td>-0.03346051</td><td> 1.050289950</td><td>1.0038981</td><td>3709.977</td></tr>
	<tr><th scope=row>a[46]</th><td>-0.63422062</td><td>0.3534991</td><td>-1.23101430</td><td>-0.048508704</td><td>1.0033814</td><td>5914.009</td></tr>
	<tr><th scope=row>a[47]</th><td> 1.89916617</td><td>0.4916798</td><td> 1.16933495</td><td> 2.721762950</td><td>1.0074324</td><td>5427.145</td></tr>
	<tr><th scope=row>a[48]</th><td>-0.05647789</td><td>0.3286388</td><td>-0.58419263</td><td> 0.469143990</td><td>1.0000265</td><td>5573.941</td></tr>
</tbody>
</table>


    
![png](chapter_13_images/output_5_1.png)
    


So far this is just a very standard model with nothing new in it. If we wanted the probability of survival we would take the logit of the `a` values, as we've done in the past.

Now let's do the multilevel model. To do that, we make the prior for the `a` parameter a function of some new variable.

$$
\begin{align*}
S_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma) & \text{for $j = 1\dots 48$} \\
\bar{\alpha} &\sim \text{Normal}(0, 1.5) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$

So we now have an adaptive prior across all of our tanks which tracks in some way the average mortality.

There are two 'levels' to this model. The first is the one that looks like our usual model, and the second is the level where we track the 'global' average across tanks.

The two new parameters $\bar{\alpha}$ and $\sigma$ are termed [[Hyperparameters]] - they are parameters for parameters. Their priors are called [[Hyperpriors]].

In principle there is no limit to the number of levels we could include in the model, but often they are constrained in practice by computational complexity or the need to actually understand the model.


```R
m13.2 <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.5 seconds.

```R
compare(m13.1, m13.2)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m13.2</th><td>201.0782</td><td>7.210405</td><td> 0.00000</td><td>     NA</td><td>21.37888</td><td>0.997425352</td></tr>
	<tr><th scope=row>m13.1</th><td>212.9971</td><td>4.622617</td><td>11.91893</td><td>3.83002</td><td>24.81349</td><td>0.002574648</td></tr>
</tbody>
</table>



A few things to note:
- The multilevel model has only about 21 effective parameters (`pWAIC`) vs. 50 actual (48 intercepts + the prior parameters). This is due to the fact that the prior for each parameter 'shrinks' them toward the average, leading to a decrease in their effective number. The mean of `sigma` is about 1.6 if you calculate it. It's a [[Regularizing Prior]], but the strength of that regularization has been learned from the data.
- The multilevel model has fewer effective variables than the regular fixed model (`m13.1`), despite in reality having more variables (50 vs. 48). This is due to the effect of the aggressive regularizing prior.

Let's take a look at how our model handles the data:


```R
# extract some data
post <- extract.samples(m13.2)

# mean intercept for each tank
d$propsurv.est <- logistic(apply(post$a, 2, mean))
```


```R
plot_df <- data.frame(
    tank = d$tank,
    survival = d$propsurv,
    label = "Empirical"
)
plot_df <- rbind(plot_df,
    data.frame(
        tank = d$tank,
        survival = d$propsurv.est,
        label = "Mean Estimate"
    )
)
ggplot(plot_df) +
    geom_point(mapping = aes(tank, survival, colour = label)) +
    geom_hline(aes(yintercept = mean(inv_logit(post$a_bar)) )) +
    annotate('label', x = 42, y = 0.75, label = "Mean Survival Probability") +
    geom_vline(aes(xintercept = 16.5)) + 
    annotate('label', x = 8, y = 0, label = 'Small Tanks') +
    geom_vline(aes(xintercept = 32.5)) +
    annotate('label', x = 24, y = 0, label = 'Medium Tanks') +
    annotate('label', x = 40, y = 0, label = 'Large Tanks')
```


    
![png](chapter_13_images/output_11_0.png)
    


What does this imply about the population distribution of survival probabilities? To find out, we'll:
- plot a bunch of Gaussian distributions, one for each of the simulated samples from the posterior distribution of both $\alpha$ and $\sigma$.
- Sample a bunch of log-odds of survival for individual tanks

The result will be a posterior distribution of variation in survial in the population of tanks.


```R
# log-odds of survival
x <- seq(-3, 4, length.out = 100)
plot_df <- data.frame(x = numeric(), y = numeric(), distribution_number = integer())
for (i in 1:100) {
    mu <- post$a_bar[i]
    sigma <- post$sigma[i]
    plot_df <- rbind(plot_df, data.frame(x = x, y = dnorm(x, mu, sigma), distribution_number = i))
}
plot_df$distribution_number <- factor(plot_df$distribution_number)

ggplot(plot_df, aes(x, y, group = distribution_number)) +
    geom_line(alpha = 0.5) +
    labs(y = "Density", x = "Log-odds of survival")
```


    
![png](chapter_13_images/output_13_0.png)
    



```R
# now the actual probabilities
simulated_tanks <- rnorm(8000, post$a_bar, post$sigma)

plot_df <- data.frame(p = inv_logit(simulated_tanks))

ggplot(plot_df) +
    geom_density(aes(p), adjust = 0.1)
```


    
![png](chapter_13_images/output_14_0.png)
    


### 1.1 Varying efects and the underfitting / overfitting trade-off

> Varying intercepts are regularized estimates, byt adaptively regularized by estimating how diverse the clusters are while still estimating the features of each cluster

Multilevel models do a better job of navigating the underfitting / overfitting problem.

To see how this could be, let's pretend that the tanks are actually natural ponds. Then there are three perspectives that we could take on how to estimate the survival rate:
1. Complete pooling. This means that we assume that the ponds are all the same, with a common intercept.
1. No pooling. This means that we assume that each pond tells us nothing about the other ponds.
1. Partial pooling. This means that we use an adaptive regularizing prior, as we did earlier in this chapter.

For the first one (complete pooling), we have lots of data and so will probably generate a very precise estimate $\alpha$. However, the estimate will not fit any particular data point very well; it will underfit the data.

If you use the no pooling approach, then the estimate for each pond will be based on very little data. As a result, the mean will be overfit to the data but the variance of the estimate will be large. 

The partial pooling approach (multilevel models) threads the line between these. If there's lots of data then the result will be very precise, and if there's not then it will tend to drift toward the average of the data. This avoids both overfitting and underfitting.

To see this in action we're going to generate our own data. This will help us to verify that our models work as we expect.

### 1.2 The model

The first step is to define the model we'll be using. Basically it's the same as what we already did, but with ponds instead of tanks.

$$
\begin{align*}
    S_i &\sim \text{Binomial}(N_i, p_i) \\
    \text{logit}(p_i) &= \alpha_{\text{POND}[i]} \\
    \alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma) & \text{for $j = 1\dots 48$} \\
    \bar{\alpha} &\sim \text{Normal}(0, 1.5) \\
    \sigma &\sim \text{Exponential}(1)
\end{align*}
$$

To simulate data, we'll need to fix the values of
- $\bar{\alpha}$, the average log-odds of survival in the entire population of ponds
- $\sigma$, the standard deviation of the distribution of logg-odds of survival
- $\alpha$, a vector of individual pond intercepts
- $N_i$, the sample size from each pond

Once we have those, it''ll be simple to simulate our data!

### 1.3 Assign values to the parameters


```R
a_bar <- 1.5
sigma <- 1.5
Ni <- as.integer(rep(c(5, 10, 25, 35), each = 15))
nponds <- length(Ni) # 15 * 4 -> 60

# now we simulate the intercept for each pond
set.seed(5005)
a_pond <- rnorm(nponds, mean = a_bar, sd = sigma)

# put it in a data frame
dsim <- data.frame(
    pond = 1:nponds,
    Ni = Ni,
    true_a = a_pond
)

dsim
```


<table class="dataframe">
<caption>A data.frame: 60 × 3</caption>
<thead>
	<tr><th scope=col>pond</th><th scope=col>Ni</th><th scope=col>true_a</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td> 1</td><td> 5</td><td> 0.56673123</td></tr>
	<tr><td> 2</td><td> 5</td><td> 1.99002317</td></tr>
	<tr><td> 3</td><td> 5</td><td>-0.13775688</td></tr>
	<tr><td> 4</td><td> 5</td><td> 1.85676651</td></tr>
</tbody>
</table>


### 1.4 Simulate survivors

Now let's simulate the binomial survival process. The number of survivors is a binomial outcome with each pond having a probability of survival $p_i = \text{inv\_logit}(\alpha_i) = \frac{e^{ \alpha_i }}{1 + e^{\alpha_i}}$

```R
dsim$Si <- rbinom(nponds, prob = logistic(dsim$true_a), size=dsim$Ni)
```

### 1.5 Compute the no-pooling estimates

This is very easy - we can actually just use the data and calculate the empirical estimate: number of survivors / size of pond! Note that we could run this through a Stan model (or whatever), but the mean would be the same. We're just missing any estimate of the uncertainty, but that's ok for now.


```R
dsim$p_nopool <- dsim$Si / dsim$Ni
```

### 1.6 Compute the partial-pooling estimates

Now to fit the model to the data using partial pooling (hierarchical model). We'll use `ulam`.


```R
dat <- list(
    Si = dsim$Si,
    Ni = dsim$Ni,
    pond = dsim$pond
)

m13.3 <- ulam(
    alist(
        Si ~ dbinom(Ni, p),
        logit(p) <- a_pond[pond],
        a_pond[pond] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1)
    ),
    data = dat,
    chains = 4
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.5 seconds.
    
```R
precis(m13.3, depth=2)
```

<table class="dataframe">
<caption>A precis: 62 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a_pond[1]</th><td> 1.65988624</td><td>1.0085597</td><td> 0.15479386</td><td> 3.33835955</td><td>0.9995353</td><td>4420.683</td></tr>
	<tr><th scope=row>a_pond[2]</th><td> 2.90822959</td><td>1.3398756</td><td> 1.04712405</td><td> 5.19245595</td><td>1.0018814</td><td>3004.773</td></tr>
	<tr><th scope=row>a_pond[3]</th><td>-0.62461362</td><td>0.8600122</td><td>-1.99488080</td><td> 0.71766552</td><td>1.0013826</td><td>3480.877</td></tr>
	<tr><th scope=row>a_pond[4]</th><td> 2.88470748</td><td>1.2674096</td><td> 1.07691985</td><td> 4.96211655</td><td>1.0034323</td><td>2954.273</td></tr>
	<tr><th scope=row>a_pond[5]</th><td> 2.90283869</td><td>1.2870118</td><td> 1.04730230</td><td> 5.19048195</td><td>1.0041054</td><td>2463.419</td></tr>
	<tr><th scope=row>a_pond[6]</th><td> 2.86141924</td><td>1.2160174</td><td> 1.12579000</td><td> 4.92912070</td><td>1.0002863</td><td>2913.118</td></tr>
	<tr><th scope=row>a_pond[7]</th><td> 0.08323686</td><td>0.8532966</td><td>-1.26758355</td><td> 1.41021815</td><td>1.0039936</td><td>3894.854</td></tr>
	<tr><th scope=row>a_pond[8]</th><td> 2.86103928</td><td>1.2452787</td><td> 1.00635600</td><td> 5.08126810</td><td>1.0044143</td><td>3660.543</td></tr>
	<tr><th scope=row>a_pond[9]</th><td> 1.63214357</td><td>0.9873611</td><td> 0.18602875</td><td> 3.24833360</td><td>1.0005107</td><td>4113.089</td></tr>
	<tr><th scope=row>a_pond[10]</th><td> 1.64936922</td><td>0.9928527</td><td> 0.16837178</td><td> 3.26092935</td><td>0.9995783</td><td>3867.943</td></tr>
	<tr><th scope=row>a_pond[11]</th><td> 2.88341248</td><td>1.1840134</td><td> 1.12441140</td><td> 4.85602465</td><td>1.0017004</td><td>2423.561</td></tr>
	<tr><th scope=row>a_pond[12]</th><td> 0.07258528</td><td>0.8295170</td><td>-1.20716410</td><td> 1.39003175</td><td>1.0030982</td><td>3803.645</td></tr>
	<tr><th scope=row>a_pond[13]</th><td> 2.88679519</td><td>1.2521863</td><td> 1.01510055</td><td> 4.97860845</td><td>1.0022083</td><td>3939.425</td></tr>
	<tr><th scope=row>a_pond[14]</th><td> 2.88281649</td><td>1.2924166</td><td> 1.01573715</td><td> 5.13895545</td><td>1.0008813</td><td>3530.874</td></tr>
	<tr><th scope=row>a_pond[15]</th><td> 2.89701372</td><td>1.3297960</td><td> 0.98711284</td><td> 5.11217525</td><td>0.9994146</td><td>2911.033</td></tr>
	<tr><th scope=row>a_pond[16]</th><td> 1.56662105</td><td>0.7288709</td><td> 0.50415809</td><td> 2.84394650</td><td>1.0056684</td><td>3714.162</td></tr>
	<tr><th scope=row>a_pond[17]</th><td>-1.43375421</td><td>0.7586111</td><td>-2.71091635</td><td>-0.29662957</td><td>1.0018077</td><td>3312.875</td></tr>
	<tr><th scope=row>a_pond[18]</th><td> 1.07297851</td><td>0.6958979</td><td> 0.03872329</td><td> 2.25471110</td><td>1.0059577</td><td>5299.181</td></tr>
	<tr><th scope=row>a_pond[19]</th><td>-0.95240108</td><td>0.6786632</td><td>-2.07539120</td><td> 0.08662152</td><td>1.0016871</td><td>4327.842</td></tr>
	<tr><th scope=row>a_pond[20]</th><td> 1.54804951</td><td>0.7643490</td><td> 0.40798497</td><td> 2.77530760</td><td>1.0002527</td><td>3566.401</td></tr>
	<tr><th scope=row>a_pond[21]</th><td>-0.16060884</td><td>0.6393441</td><td>-1.20282815</td><td> 0.83028261</td><td>1.0030545</td><td>4944.697</td></tr>
	<tr><th scope=row>a_pond[22]</th><td> 2.25152715</td><td>0.9324078</td><td> 0.90781505</td><td> 3.81120310</td><td>0.9998040</td><td>3526.614</td></tr>
	<tr><th scope=row>a_pond[23]</th><td> 3.23931817</td><td>1.1356209</td><td> 1.67503440</td><td> 5.18412150</td><td>1.0013413</td><td>3348.751</td></tr>
	<tr><th scope=row>a_pond[24]</th><td> 0.62166002</td><td>0.6224451</td><td>-0.35062381</td><td> 1.62889410</td><td>1.0022992</td><td>4060.234</td></tr>
	<tr><th scope=row>a_pond[25]</th><td> 3.29010498</td><td>1.1997874</td><td> 1.62144450</td><td> 5.36935185</td><td>1.0064335</td><td>3158.397</td></tr>
	<tr><th scope=row>a_pond[26]</th><td> 2.22989451</td><td>0.8996433</td><td> 0.90760689</td><td> 3.77986180</td><td>0.9999014</td><td>3454.550</td></tr>
	<tr><th scope=row>a_pond[27]</th><td> 1.05548051</td><td>0.6699028</td><td> 0.03753697</td><td> 2.15367505</td><td>0.9999907</td><td>4324.304</td></tr>
	<tr><th scope=row>a_pond[28]</th><td> 2.23254095</td><td>0.8734175</td><td> 0.94810737</td><td> 3.71547465</td><td>1.0082535</td><td>4251.251</td></tr>
	<tr><th scope=row>a_pond[29]</th><td> 1.57638561</td><td>0.7388113</td><td> 0.45475913</td><td> 2.79557285</td><td>0.9994820</td><td>3668.306</td></tr>
	<tr><th scope=row>a_pond[30]</th><td> 1.04288704</td><td>0.6761319</td><td> 0.03467851</td><td> 2.16845875</td><td>1.0005339</td><td>3686.254</td></tr>
	<tr><th scope=row>⋮</th><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><th scope=row>a_pond[33]</th><td> 1.729826671</td><td>0.5285802</td><td> 0.95513611</td><td> 2.6231067</td><td>1.0022817</td><td>3913.9272</td></tr>
	<tr><th scope=row>a_pond[34]</th><td> 1.244389024</td><td>0.4728283</td><td> 0.50956235</td><td> 2.0052454</td><td>1.0018902</td><td>3688.0856</td></tr>
	<tr><th scope=row>a_pond[35]</th><td> 0.661364881</td><td>0.4089161</td><td> 0.01510982</td><td> 1.3181901</td><td>1.0026253</td><td>4230.4677</td></tr>
	<tr><th scope=row>a_pond[36]</th><td> 3.837529060</td><td>1.0336977</td><td> 2.34673250</td><td> 5.6492970</td><td>1.0001253</td><td>2437.2189</td></tr>
	<tr><th scope=row>a_pond[37]</th><td>-0.988240077</td><td>0.4346930</td><td>-1.71186035</td><td>-0.3210750</td><td>1.0012422</td><td>3646.7817</td></tr>
	<tr><th scope=row>a_pond[38]</th><td>-1.192465540</td><td>0.4722272</td><td>-1.97269830</td><td>-0.4703791</td><td>1.0080495</td><td>4510.8249</td></tr>
	<tr><th scope=row>a_pond[39]</th><td> 0.667472885</td><td>0.4310774</td><td> 0.02278333</td><td> 1.3646030</td><td>1.0026915</td><td>3792.0056</td></tr>
	<tr><th scope=row>a_pond[40]</th><td> 3.852422695</td><td>1.0665439</td><td> 2.34982260</td><td> 5.7361128</td><td>1.0031673</td><td>2689.7708</td></tr>
	<tr><th scope=row>a_pond[41]</th><td> 3.852268495</td><td>1.0889751</td><td> 2.36126405</td><td> 5.7819006</td><td>1.0006275</td><td>3060.2526</td></tr>
	<tr><th scope=row>a_pond[42]</th><td> 2.466787424</td><td>0.6989907</td><td> 1.46877350</td><td> 3.6379816</td><td>1.0031976</td><td>4273.7207</td></tr>
	<tr><th scope=row>a_pond[43]</th><td>-0.127785952</td><td>0.3882158</td><td>-0.74288822</td><td> 0.4955991</td><td>1.0028909</td><td>4971.5770</td></tr>
	<tr><th scope=row>a_pond[44]</th><td> 0.673961175</td><td>0.3874305</td><td> 0.06019406</td><td> 1.2960422</td><td>1.0035083</td><td>4928.0808</td></tr>
	<tr><th scope=row>a_pond[45]</th><td>-1.190294806</td><td>0.4742464</td><td>-1.99217330</td><td>-0.4588302</td><td>0.9990809</td><td>4813.6941</td></tr>
	<tr><th scope=row>a_pond[46]</th><td> 0.004155066</td><td>0.3297293</td><td>-0.53088607</td><td> 0.5316434</td><td>1.0003552</td><td>4194.2279</td></tr>
	<tr><th scope=row>a_pond[47]</th><td> 4.065463275</td><td>1.0949588</td><td> 2.54808875</td><td> 5.9144301</td><td>1.0013355</td><td>2475.4901</td></tr>
	<tr><th scope=row>a_pond[48]</th><td> 2.085699492</td><td>0.5191137</td><td> 1.29768890</td><td> 2.9558292</td><td>1.0000303</td><td>3798.3450</td></tr>
	<tr><th scope=row>a_pond[49]</th><td> 1.850587844</td><td>0.4686794</td><td> 1.14700325</td><td> 2.6185046</td><td>1.0079853</td><td>4130.5650</td></tr>
	<tr><th scope=row>a_pond[50]</th><td> 2.746846295</td><td>0.6257955</td><td> 1.82925445</td><td> 3.8250139</td><td>1.0011958</td><td>3564.3574</td></tr>
	<tr><th scope=row>a_pond[51]</th><td> 2.393873983</td><td>0.5684531</td><td> 1.53779710</td><td> 3.3464912</td><td>1.0084080</td><td>3890.1543</td></tr>
	<tr><th scope=row>a_pond[52]</th><td> 0.365686217</td><td>0.3188737</td><td>-0.13196777</td><td> 0.8675500</td><td>1.0004398</td><td>3878.7595</td></tr>
	<tr><th scope=row>a_pond[53]</th><td> 2.096703591</td><td>0.5009239</td><td> 1.35011910</td><td> 2.9293713</td><td>1.0031608</td><td>4417.1300</td></tr>
	<tr><th scope=row>a_pond[54]</th><td> 4.105577515</td><td>1.0642939</td><td> 2.60252205</td><td> 5.9793212</td><td>1.0014540</td><td>2537.0765</td></tr>
	<tr><th scope=row>a_pond[55]</th><td> 1.127769104</td><td>0.3844349</td><td> 0.55028251</td><td> 1.7316130</td><td>1.0016196</td><td>4922.2004</td></tr>
	<tr><th scope=row>a_pond[56]</th><td> 2.779257217</td><td>0.6432820</td><td> 1.84219970</td><td> 3.8741420</td><td>1.0015164</td><td>3228.7223</td></tr>
	<tr><th scope=row>a_pond[57]</th><td> 0.720788883</td><td>0.3749139</td><td> 0.12404182</td><td> 1.3114648</td><td>1.0033347</td><td>3384.2285</td></tr>
	<tr><th scope=row>a_pond[58]</th><td> 4.069900565</td><td>1.0436149</td><td> 2.61679555</td><td> 5.8561631</td><td>1.0018504</td><td>2478.0483</td></tr>
	<tr><th scope=row>a_pond[59]</th><td> 1.635110636</td><td>0.4381356</td><td> 0.96118998</td><td> 2.3679203</td><td>1.0074139</td><td>3876.5574</td></tr>
	<tr><th scope=row>a_pond[60]</th><td> 2.389304027</td><td>0.5715276</td><td> 1.56573435</td><td> 3.3522229</td><td>1.0024700</td><td>4124.2886</td></tr>
	<tr><th scope=row>a_bar</th><td> 1.663679393</td><td>0.2437952</td><td> 1.28092130</td><td> 2.0674074</td><td>1.0002220</td><td>1608.8907</td></tr>
	<tr><th scope=row>sigma</th><td> 1.670055920</td><td>0.2458333</td><td> 1.31172670</td><td> 2.0957813</td><td>1.0016408</td><td> 884.8431</td></tr>
</tbody>
</table>


Let's compute the predicted survival proportions and add them to the data frame.


```R
post <- extract.samples(m13.3)
dsim$p_partpool <- apply(inv_logit(post$a_pond), 2, mean)

# we'll also need the true p value for each pond
dsim$p_true <- inv_logit(dsim$true_a)

# no we calculate the error for each of the different methods
nopool_error <- abs(dsim$p_nopool - dsim$p_true)
partpool_error <- abs(dsim$p_partpool - dsim$p_true)
```


```R
plot_df <- rbind(
    data.frame(
        tank = 1:length(nopool_error),
        error = nopool_error,
        type = "No Pooling"
    ),
    data.frame(
        tank = 1:length(partpool_error),
        error = partpool_error,
        type = "Partial Pooling"
    )
)

ggplot(plot_df, aes(tank, error, colour = type)) +
    geom_point() +
    labs(x = "Tank", y = "Absolute Error", colour = "Type") 

```


    
![png](chapter_13_images/output_26_0.png)
    


Note that the partial pooling results are better almost every time than the no pooling results.


```R
nopool_avg <- aggregate(nopool_error, list(dsim$Ni), mean)
partpool_avg <- aggregate(partpool_error, list(dsim$Ni), mean)

nopool_avg
partpool_avg
```


<table class="dataframe">
<caption>A data.frame: 4 × 2</caption>
<thead>
	<tr><th scope=col>Group.1</th><th scope=col>x</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td> 5</td><td>0.12122935</td></tr>
	<tr><td>10</td><td>0.10245961</td></tr>
	<tr><td>25</td><td>0.04057566</td></tr>
	<tr><td>35</td><td>0.03583510</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A data.frame: 4 × 2</caption>
<thead>
	<tr><th scope=col>Group.1</th><th scope=col>x</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td> 5</td><td>0.08512961</td></tr>
	<tr><td>10</td><td>0.07655761</td></tr>
	<tr><td>25</td><td>0.04837656</td></tr>
	<tr><td>35</td><td>0.03616521</td></tr>
</tbody>
</table>



## 2 More than one type of cluster

It often turns out that our experimental structure lends itself to more than one type of cluster. For instance, in the chimpanzee experiment, each level pulls belongs to one actor (the chimpanzee), but the pulls *also* happened within discrete experimental blocks, each occurring on the same day. Each pull belongs to an actor (1 - 7) but also to a block (1 - 6). We can use partial pooling on both of these clusters.

This kind of data, where e.g. actors are not nested within a single block, is called a [[Cross-Classified Multilevel Model]]. In each actor had done all of their pulls within the same blocks, then it would be a [[Hierarchical Multilevel Model]]. The model specification is typically the same between these.

### 2.1 Multilevel chimpanzees

To adapt for both the actor-level and block-level clustering, we'll adapt the model from [[Chapter 11]] and add a new actor-level and block-level intercept, each with their own parameters.

$$
\begin{align*}
L_i &\sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{ACTOR}[i]} + \gamma_{\text{BLOCK}[i]} + \beta_{\text{TREATMENT}[i]} \\
\beta_j &\sim \text{Normal}(0, 0.5) & \text{for $j = 1\dots 4$} \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma_{\alpha}) & \text{for $j = 1\dots 7$} \\
\gamma_j &\sim \text{Normal}(0, \sigma_{\gamma}) & \text{for $j = 1\dots 6$} \\
\bar{\alpha} &\sim \text{Normal(0, 1.5)} \\
\sigma_\alpha &\sim \text{Exponential(1)} \\
\sigma_\gamma &\sim \text{Exponential(1)} \\
\end{align*}
$$

Note that there's only one $\bar{\alpha}$ parameter. We can't identify a separate mean for each varying intercept type because both intercepts are added to the same prediction. In theory we could add separate ones, but then we'd run into a multicollinearity problem like we did earlier with the left and right legs example.

Now let's run the model!


```R
data(chimpanzees)
d <- chimpanzees
d$treatment <- 1 + d$prosoc_left + 2 * d$condition

dat_list <- list(
    pulled_left = d$pulled_left,
    actor = d$actor,
    block_id = d$block,
    treatment = as.integer(d$treatment)
)

set.seed(13)
m13.4 <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + g[block_id] + b[treatment],
        b[treatment] ~ dnorm(0, 0.5),
        # adaptive priors
        a[actor] ~ dnorm(a_bar, sigma_a),
        g[block_id] ~ dnorm(0, sigma_g),
        # hyper priors
        a_bar ~ dnorm(0, 1.5),
        sigma_a ~ dexp(1),
        sigma_g ~ dexp(1)
    ),
    data = dat_list,
    chains = 4,
    cores = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 1.1 seconds.
    Total execution time: 1.4 seconds.

    Warning: 5 of 2000 (0.0%) transitions ended with a divergence.
    See https://mc-stan.org/misc/warnings for details.

Apparently this is supposed to result in a bunch of divergent transitions, but it's not at the moment. Apparently this is something that we'll fix in a bit.

```R
par(bg = 'white')
precis(m13.4, depth = 2)
plot(precis(m13.4, depth = 2))
```


<table class="dataframe">
<caption>A precis: 20 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>b[1]</th><td>-0.140384396</td><td>0.3104392</td><td>-0.656763650</td><td> 0.340089770</td><td>1.010293</td><td> 438.7808</td></tr>
	<tr><th scope=row>b[2]</th><td> 0.393819689</td><td>0.3105997</td><td>-0.111781670</td><td> 0.886487385</td><td>1.012343</td><td> 400.4837</td></tr>
	<tr><th scope=row>b[3]</th><td>-0.478942576</td><td>0.3125054</td><td>-0.985714350</td><td>-0.001867275</td><td>1.014307</td><td> 426.3076</td></tr>
	<tr><th scope=row>b[4]</th><td> 0.281226117</td><td>0.3040646</td><td>-0.214599940</td><td> 0.738879970</td><td>1.014526</td><td> 465.1455</td></tr>
	<tr><th scope=row>a[1]</th><td>-0.349393397</td><td>0.3709872</td><td>-0.915795145</td><td> 0.265304750</td><td>1.008104</td><td> 414.0663</td></tr>
	<tr><th scope=row>a[2]</th><td> 4.693184815</td><td>1.2253668</td><td> 3.033960000</td><td> 6.876075550</td><td>1.003570</td><td> 799.5012</td></tr>
	<tr><th scope=row>a[3]</th><td>-0.656808101</td><td>0.3687594</td><td>-1.217484700</td><td>-0.057030201</td><td>1.008510</td><td> 388.3045</td></tr>
	<tr><th scope=row>a[4]</th><td>-0.656701408</td><td>0.3752144</td><td>-1.240150050</td><td>-0.072011782</td><td>1.008084</td><td> 404.3305</td></tr>
	<tr><th scope=row>a[5]</th><td>-0.354013953</td><td>0.3711532</td><td>-0.920823360</td><td> 0.261819015</td><td>1.010378</td><td> 423.3406</td></tr>
	<tr><th scope=row>a[6]</th><td> 0.594438954</td><td>0.3779616</td><td> 0.007510246</td><td> 1.203294600</td><td>1.008408</td><td> 388.7268</td></tr>
	<tr><th scope=row>a[7]</th><td> 2.121230156</td><td>0.4672240</td><td> 1.387656900</td><td> 2.872632750</td><td>1.007484</td><td> 366.7395</td></tr>
	<tr><th scope=row>g[1]</th><td>-0.177523192</td><td>0.2340502</td><td>-0.606642055</td><td> 0.068304781</td><td>1.006124</td><td> 546.3455</td></tr>
	<tr><th scope=row>g[2]</th><td> 0.026140733</td><td>0.1804243</td><td>-0.245996270</td><td> 0.318646150</td><td>1.008799</td><td> 949.5756</td></tr>
	<tr><th scope=row>g[3]</th><td> 0.052752477</td><td>0.1865732</td><td>-0.199796370</td><td> 0.385483410</td><td>1.011404</td><td> 711.8376</td></tr>
	<tr><th scope=row>g[4]</th><td> 0.009976589</td><td>0.1813646</td><td>-0.274479290</td><td> 0.289173435</td><td>1.006694</td><td>1071.4778</td></tr>
	<tr><th scope=row>g[5]</th><td>-0.032283694</td><td>0.1822856</td><td>-0.348219435</td><td> 0.230681665</td><td>1.012738</td><td> 946.1959</td></tr>
	<tr><th scope=row>g[6]</th><td> 0.106378586</td><td>0.1978929</td><td>-0.138192100</td><td> 0.465116445</td><td>1.001676</td><td> 818.4058</td></tr>
	<tr><th scope=row>a_bar</th><td> 0.606644243</td><td>0.7361489</td><td>-0.520919780</td><td> 1.800580700</td><td>1.002886</td><td>1051.5382</td></tr>
	<tr><th scope=row>sigma_a</th><td> 2.031491420</td><td>0.6466113</td><td> 1.205757900</td><td> 3.259981250</td><td>1.003725</td><td> 781.6992</td></tr>
	<tr><th scope=row>sigma_g</th><td> 0.216229703</td><td>0.1768724</td><td> 0.028509086</td><td> 0.541332890</td><td>1.019222</td><td> 195.4716</td></tr>
</tbody>
</table>




    
![png](chapter_13_images/output_33_1.png)
    


There are a few things that we should notice about this.

- The effective number of parameters (`ess_bulk`) varies a lot between the different parameters. There are a lot of reasons for this, but in out case a large part of the reason is that some parameters spend a lot of time near a boundary. In this case, it's `sigma_g`. Also a bunch of the `r_hat` values are above one. Both of these are signs of inefficient sampling, which we'll fix later.
- Notice that `sigma_g` (sd of the blocks) is much smaller than `sigma_a`, the variance among the actors. If we look at the values for these (`a[]` and `g[]`) this makes sense - the blocks all seem to be basically the same while there's a lot of variation among the actors.

As a result, adding `block` to this model hasn't added a lot of overfitting risk. Let's take a look at the varying intercepts model without the blocks.


```R
set.seed(14)
m13.5 <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + b[treatment],
        b[treatment] ~ dnorm(0, 0.5),
        # adaptive priors
        a[actor] ~ dnorm(a_bar, sigma_a),
        # hyper priors
        a_bar ~ dnorm(0, 1.5),
        sigma_a ~ dexp(1)
    ),
    data = dat_list,
    chains = 4,
    cores = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.6 seconds.
    Total execution time: 0.7 seconds.


```R
compare(m13.4, m13.5)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m13.5</th><td>531.5236</td><td>19.24054</td><td>0.0000000</td><td>      NA</td><td> 8.760136</td><td>0.5637934</td></tr>
	<tr><th scope=row>m13.4</th><td>532.0367</td><td>19.39689</td><td>0.5131437</td><td>1.707225</td><td>10.593521</td><td>0.4362066</td></tr>
</tbody>
</table>



These models are basically identical in terms of their efficacy.

Note that it is tempting to just coose m13.4 as the model to use and go from there. However, that risks us losing valuable information - we now know something that *didn't* really affect the model! It's probably a good practice to report both and show what happened.

### 2.2 Even more clusters

Notice that the treatment effects, `b`, look like the `a` and `g` parameters. Could we also partially pool these? Absolutely!

Note that some people have been taught that you should only ever pool things that were not experimentally controlled (varying effects are only for these variables). The idea is that since the treatment was 'fixed' by the experiment, we should use un-pooled 'fixed' effects.

This is not correct. We use varying effects (pooling) because they provide better inferences. It doesn't matter how the clusters are created. What's important is that the individual units are **exhangable** - that is, we could swap the individual index values without changing the meaning of the model.

In this case there are only four treatments and there is a lot of data on each treatment, so it probably won't make a different anyway. Let's test it out!


```R
set.seed(15)
m13.6 <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + g[block_id] + b[treatment],
        # adaptive priors
        a[actor] ~ dnorm(a_bar, sigma_a),
        g[block_id] ~ dnorm(0, sigma_g),
        b[treatment] ~ dnorm(0, sigma_b),
        # hyper priors
        a_bar ~ dnorm(0, 1.5),
        sigma_a ~ dexp(1),
        sigma_g ~ dexp(1),
        sigma_b ~ dexp(1)
    ),
    data = dat_list,
    chains = 4,
    cores = 4,
    log_lik = TRUE
)
coeftab(m13.4, m13.6)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 1.0 seconds.
    Total execution time: 1.1 seconds.
    
    Warning: 2 of 2000 (0.0%) transitions ended with a divergence.
    See https://mc-stan.org/misc/warnings for details.
    
            m13.4   m13.6  
    b[1]      -0.14   -0.09
    b[2]       0.39    0.43
    b[3]      -0.48   -0.42
    b[4]       0.28    0.31
    a[1]      -0.35   -0.39
    a[2]       4.69    4.65
    a[3]      -0.66   -0.70
    a[4]      -0.66   -0.70
    a[5]      -0.35   -0.39
    a[6]       0.59    0.54
    a[7]       2.12    2.08
    g[1]      -0.18   -0.18
    g[2]       0.03    0.03
    g[3]       0.05    0.04
    g[4]       0.01    0.01
    g[5]      -0.03   -0.04
    g[6]       0.11    0.11
    a_bar      0.61    0.56
    sigma_a    2.03    2.02
    sigma_g    0.22    0.22
    sigma_b      NA    0.61
    nobs        504     504


If we look just at the `b` parameters, we see that they're basically the same. The variable `sigma_b` is very small. Basically, the treatments don't vary a lot because they don't really do that much. Since there's a lot of data on each treatment, they don't get pooled much anyway. This is broadly typical when each cluster has a lot of data.

However, we do get more divergent transitions. We'll look at how to fix those in a bit.

## 3 Divergent transitions and non-centred priors

The previous models were supposed to produce divergent transitions (but tragically didn't - we'll continue as though they did).

[[Divergent transitions]] are commonplace when working with multilevel models, and so it's important to know how to fix them.

In [[Hamiltonian Monte Carlo]], the idea is that the particle is given a 'flick', and then we track its position to give us the different samples from the posterior. In principle, the total energy of the system should be the same at the start and the end. However, due to the vagaries of numerical approximations, sometimes it is not! This is then called a [[divergent transition]].

When does this tend to happen? Usually, when the posterior distribution is very steep in some region of the parameter space. These steep changes are hard for a discrete numerical approximation to ... approximate.

These divergent transitions are rejected, so they don't directly hurt the simulation. However, they do hurt it indirectly because those steep regions should still be explored - they're still part of the posterior!

There are two main ways to adjust for these divergent transitions:
1. Tune to simulation so that it doesn't overshoot. In [[Stan]], this involves doing more warmup with a higher target acceptance rate (`adapt_delta`). Note that for some models, they can't be fixed using this method - you can never adjust it enough to get rid of all of the divergent transitions.
2. [[Reparameterize]] it. For any model, it can be written in several different equivalent forms.

Let's take a look at two examples.

### 3.1 The Devil's Funnel

You don't need a fancy model to produce divergent transitions:

$$
\begin{align*}
v &\sim \text{Normal}(0, 3) \\
x &\sim \text{Normal}(0, e^v) \\
\end{align*}
$$


```R
m13.7 <- ulam(
    alist(
        v ~ normal(0, 3),
        x ~ normal(0, exp(v))
    ),
    data = list(N = 1),
    chains = 4
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.

    Warning: 88 of 2000 (4.0%) transitions ended with a divergence.
    See https://mc-stan.org/misc/warnings for details.


OK, so now there are some divergent transitions!


```R
traceplot(m13.7)
```


    Error: unable to find an inherited method for function ‘traceplot’ for signature ‘object = "ulam"’
    Traceback:


    1. traceplot(m13.7)

    2. (function (classes, fdef, mtable) 
     . {
     .     methods <- .findInheritedMethods(classes, fdef, mtable)
     .     if (length(methods) == 1L) 
     .         return(methods[[1L]])
     .     else if (length(methods) == 0L) {
     .         cnames <- paste0(fdef@signature[seq_along(classes)], 
     .             " = \"", vapply(classes, as.character, ""), "\"", 
     .             collapse = ", ")
     .         stop(gettextf("unable to find an inherited method for function %s for signature %s", 
     .             sQuote(fdef@generic), sQuote(cnames)), call. = FALSE, 
     .             domain = NA)
     .     }
     .     else stop("Internal error in finding inherited methods; didn't return a unique method", 
     .         domain = NA)
     . })(list(structure("ulam", package = "rethinking")), new("nonstandardGenericFunction", 
     .     .Data = function (object, ...) 
     .     {
     .         standardGeneric("traceplot")
     .     }, generic = structure("traceplot", package = "rstan"), package = "rstan", 
     .     group = list(), valueClass = character(0), signature = "object", 
     .     default = NULL, skeleton = (function (object, ...) 
     .     stop(gettextf("invalid call in method dispatch to '%s' (no default method)", 
     .         "traceplot"), domain = NA))(object, ...)), <environment>)

    3. stop(gettextf("unable to find an inherited method for function %s for signature %s", 
     .     sQuote(fdef@generic), sQuote(cnames)), call. = FALSE, domain = NA)


Yikes! This doesn't look like a spiky caterpillar at all!

What's happening here is that around zero, the entire distribution contracts, creating a very steep valley of probability density. The actual simulation will then overshoot this valley, leading to the ivergent transitions.

For us, it's the dependence of $x$ on $v$ which creates the problem:

$$
x \sim \text{Normal}(0, exp(v))
$$

as $v$ changes, the distribution of $x$ changes in a very inconvenient way.

This parameterization is known as the [[Centred Parameterization]]. It just indicates that the distribution of $x$ is conditional on one or more other parameters.

The alternative is a [[Non-centred Parameterization]]. This is one where the embedded parameter, $v$, is moved out of the definition of the other parameter. For us, this might look like

$$
\begin{align*}
v &\sim \text{Normal}(0, 3) \\
z &\sim \text{Normal}(0, 1) \\
x &= z \ast \exp (v)
\end{align*}
$$

What's going on here? Basically, we're 'unstandardizing' $z$ to get $x$. Normally when standardizing a variable we subtract the mean and divide by the standard deviation to get something with mean 0 and sd 1. The new variable $z$ is the standardized form of $x$: $z = \frac{x - 0}{\exp (v)} \to x = z \ast \exp (v)$. So to get $x$ back, we just undo the standardization.

Now when we run the Markov chain, we sample from $z$, not $x$.


```R
m13.7nc <- ulam(
    alist(
        v ~ normal(0, 3),
        z ~ normal(0, 1),
        gq > real[1]:x <<- z * exp(v)
    ),
    data = list(N = 1),
    chains = 4
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.

Now we get no divergent transitions!

```R
precis(m13.7nc)
```

<table class="dataframe">
<caption>A precis: 3 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>v</th><td>-0.186937548</td><td>  3.0302187</td><td> -5.009651</td><td> 4.630757</td><td>1.001919</td><td>1447.016</td></tr>
	<tr><th scope=row>z</th><td> 0.003852076</td><td>  0.9980279</td><td> -1.582904</td><td> 1.606554</td><td>1.001535</td><td>1391.508</td></tr>
	<tr><th scope=row>x</th><td> 4.050962936</td><td>382.8535941</td><td>-19.955182</td><td>19.116465</td><td>1.000364</td><td>1447.043</td></tr>
</tbody>
</table>



So we managed to reparameterize this model and sample from the variable that we wanted by sampling a different variable and then transforming it.

### 3.2 Non-centred chimpanzees

In the chimpanzee model, the adaptive priors that make it a multilevel model also cause regions of steep curvature and hence divergent transitions. We'd like to fix that!

Before we reparameterize, maybe we can get away with changing the `adapt_delta` parameter in Stan. The default which `ulam` uses is 0.95, which means that it aims for a 95% acceptance rate. During the warmup phase, it uses this to adjust the step size of each [[leapfrog step]] When `adapt_delta` is high it results in smaller steps, giving a more detailed approzimation of the posterior.

Increasing `adapt_delta` will often, but not always, help with divergent transitions.

Let's see what happens if we run model `m13.4` with a higher target acceptance rate!


```R
# original version
m13.4a <- ulam(m13.4, chains = 4, cores = 4, control = list(adapt_delta = 0.95))
divergent(m13.4a)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.7 seconds.
    Total execution time: 0.9 seconds.

    Warning: 3 of 2000 (0.0%) transitions ended with a divergence.
    See https://mc-stan.org/misc/warnings for details.

3
```R
# increasing the acceptance rate
m13.4b <- ulam(m13.4, chains = 4, cores = 4, control = list(adapt_delta = 0.99))
divergent(m13.4b)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 1.9 seconds.
    Total execution time: 2.8 seconds.

0

So this is a slight improvement. Note that even though we have fewer divergent transitions, it's still not great - the chain isn't very efficient. If we look at the precis, we see that the effective number of samples is far below the actual number (2000):

(Actually we're looking for `n_eff`, which isn't showing up. But apparently it's low!)


```R
precis(m13.4b, depth = 2)
```


<table class="dataframe">
<caption>A precis: 20 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>b[1]</th><td>-0.135335638</td><td>0.2986754</td><td>-0.611329045</td><td> 0.34835652</td><td>1.0067707</td><td> 654.2604</td></tr>
	<tr><th scope=row>b[2]</th><td> 0.391353318</td><td>0.2982066</td><td>-0.073981611</td><td> 0.86401174</td><td>1.0040832</td><td> 612.5395</td></tr>
	<tr><th scope=row>b[3]</th><td>-0.473586060</td><td>0.3012208</td><td>-0.941730935</td><td> 0.01350109</td><td>1.0013559</td><td> 666.8144</td></tr>
	<tr><th scope=row>b[4]</th><td> 0.279498288</td><td>0.2890972</td><td>-0.189350110</td><td> 0.72798494</td><td>1.0038989</td><td> 614.9519</td></tr>
	<tr><th scope=row>a[1]</th><td>-0.349938270</td><td>0.3687439</td><td>-0.941289805</td><td> 0.20651201</td><td>1.0035689</td><td> 497.7386</td></tr>
	<tr><th scope=row>a[2]</th><td> 4.623227610</td><td>1.2407716</td><td> 3.005832800</td><td> 6.80323905</td><td>0.9996655</td><td>1289.4706</td></tr>
	<tr><th scope=row>a[3]</th><td>-0.665077759</td><td>0.3759564</td><td>-1.271050000</td><td>-0.07712462</td><td>1.0056842</td><td> 532.5694</td></tr>
	<tr><th scope=row>a[4]</th><td>-0.667970706</td><td>0.3691988</td><td>-1.253683300</td><td>-0.09762715</td><td>1.0022536</td><td> 586.3560</td></tr>
	<tr><th scope=row>a[5]</th><td>-0.350192426</td><td>0.3548896</td><td>-0.903757570</td><td> 0.20699759</td><td>1.0046514</td><td> 533.7358</td></tr>
	<tr><th scope=row>a[6]</th><td> 0.581713052</td><td>0.3687691</td><td>-0.000324762</td><td> 1.19082240</td><td>1.0050519</td><td> 631.3033</td></tr>
	<tr><th scope=row>a[7]</th><td> 2.114819485</td><td>0.4779974</td><td> 1.339897300</td><td> 2.89870805</td><td>1.0031557</td><td> 776.2180</td></tr>
	<tr><th scope=row>g[1]</th><td>-0.157534358</td><td>0.2158190</td><td>-0.556610840</td><td> 0.07243119</td><td>1.0070861</td><td> 596.2390</td></tr>
	<tr><th scope=row>g[2]</th><td> 0.039811591</td><td>0.1829950</td><td>-0.216339460</td><td> 0.35750533</td><td>1.0143775</td><td> 794.8721</td></tr>
	<tr><th scope=row>g[3]</th><td> 0.051019052</td><td>0.1941289</td><td>-0.206049805</td><td> 0.40323823</td><td>1.0089068</td><td> 747.4024</td></tr>
	<tr><th scope=row>g[4]</th><td> 0.009120271</td><td>0.1822555</td><td>-0.257563820</td><td> 0.30012466</td><td>1.0231992</td><td> 891.1370</td></tr>
	<tr><th scope=row>g[5]</th><td>-0.025518160</td><td>0.1873838</td><td>-0.315726095</td><td> 0.24533179</td><td>1.0148384</td><td> 925.2087</td></tr>
	<tr><th scope=row>g[6]</th><td> 0.109728740</td><td>0.1990603</td><td>-0.118562640</td><td> 0.47718371</td><td>1.0120828</td><td> 571.5213</td></tr>
	<tr><th scope=row>a_bar</th><td> 0.604395227</td><td>0.7355361</td><td>-0.582012605</td><td> 1.75500855</td><td>1.0034309</td><td> 899.6368</td></tr>
	<tr><th scope=row>sigma_a</th><td> 1.992167091</td><td>0.6443210</td><td> 1.201411850</td><td> 3.14037540</td><td>1.0015380</td><td>1190.8014</td></tr>
	<tr><th scope=row>sigma_g</th><td> 0.207995755</td><td>0.1723785</td><td> 0.022268320</td><td> 0.52272063</td><td>1.0340488</td><td> 126.8770</td></tr>
</tbody>
</table>



What we really want is a non-centred version of the model. That is, we need to get the parameters out of the adaptive priors and into the linear model. There are two adaptive priors to transform:

$$
\begin{align*}
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma_\alpha) \\
\gamma_j &\sim \text{Normal}(0, \sigma_\gamma)
\end{align*}
$$

Within these, there are three 'centred' parameters that we need to remove from the priors: $\bar{\alpha}$, $\sigma_\alpha$, and $\sigma\gamma$. Just like with the earlier Funnel problem, we'll defined new variables that are given standard Normal distributions and then reconstruct the original variables by undoing the transformation.

This time, we'll do it in the linear model.

$$
\begin{align*}
L_i &\sim \text{Binomial}(1, p) \\
\text{logit}(p) &= \underbrace{\bar\alpha + z_{\text{ACTOR}[i]}\sigma_\alpha}_{\alpha_{\text{ACTOR}[i]}} + \underbrace{x_{\text{BLOCK}[i]}\sigma_\gamma}_{\gamma_{\text{BLOCK}[i]}} + \beta_{\text{TREATMENT}[i]} \\
\beta_j &\sim \text{Normal}(0, 0.5) & \text{for $j=1\dots 4$} \\
z_j &\sim \text{Normal}(0, 1) \\
x_j &\sim \text{Normal}(0, 1) \\
\bar\alpha &\sim \text{Normal}(0, 1.5) \\
\sigma_\alpha &\sim \text{Exponential}(1) \\
\sigma_\gamma &\sim \text{Exponential}(1) \\
\end{align*}
$$

The vector $z$ gives the standardized intercept for each actor, and the vector $x$ gives the standardized intercept for each block. We've reparameterized our old variables as

$$
\begin{align*}
\alpha_j &= \bar\alpha + z_j \sigma_alpha \\
\gamma_j &= x_j \sigma_\gamma \\
\end{align*}
$$


```R
m13.4nc <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a_bar + z[actor] * sigma_a + x[block_id] * sigma_g + b[treatment],
        b[treatment] ~ dnorm(0, 0.5),
        z[actor] ~ dnorm(0, 1),
        x[block_id] ~ dnorm(0, 1),
        a_bar ~ dnorm(0, 1.5),
        sigma_a ~ dexp(1),
        sigma_g ~ dexp(1),
        gq> vector[actor]:a <<- a_bar + z * sigma_a,
        gq> vector[block_id]:g <<- x * sigma_g
    ),
    data = dat_list,
    chains = 4,
    cores = 4
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 1.5 seconds.
    Total execution time: 1.8 seconds.
    
Now let's compare the `n_eff` for each of these!

```R
precis_c <- precis(m13.4, depth = 2)
precis_nc <- precis(m13.4nc, depth = 2)

pars <- c(
    paste("a[", 1:7, "]", sep = ""),
    paste("g[", 1:6, "]", sep = ""),
    paste("b[", 1:4, "]", sep = ""),
    "a_bar", "sigma_a", "sigma_g"
)
# precis_c[, "n_eff"] # results in an error

neff_table <- cbind(precis_c[pars, "n_eff"], precis_nc[pars, "n_eff"])
# for some reason precis no longer has these columns, so none of this works...
```

Since `precis` no longer has the `n_eff` column, we can't really compare. But suffice it to say that the non-centred version generally has a larger `n_eff`, meaning that the sampling was more efficient.

So, should we always just use a non-centred version of the model? No! There are time when each kind (centred vs. non-centred) work better. Sometimes one version works better for one cluster and the other work better for a different cluster! Generally
- A cluster with low variation, like the block in this model, will sample better with a non-centred prior
- If you have a large number of units within the cluster but not much data for each unit, then the non-centred is usually better
- Otherwise, probably the centred one will work best

We can also reparameterize distributions other than the Gaussian. For instance, if we have an exponential, then

$$
\begin{align*}
    x &= z\lambda \\
    z &\sim \text{Exponential}(1)
\end{align*}
$$
is equivalent to $\text{Exponential}(\lambda)$. In [[SR - Chapter 14 - Adventures in Covariance]], we'll look at how we can reparameterize multivariate distributions so as to place an entire correlation matrix inside a linear model.

## 4 Multilevel posterior predictions

It's important to check your models! One of the best ways to do this is to explore the predictions that your model makes.

Once you think that the posterior is correct, then you can use the predictions to explore the causal effects. What does your model predict will happen when you alter one of the variables?

We can also use this to compute [[Information Criteria]] like [[AIC]] or [[WAIC]].

All of this advice also applies to multilevel models. However, the introduction of varying effects does introduce nuance.

First, we should no longer expect the model to exactly retrodict the sample; adaptive regularization has as its goal the trade of pooer fit in sample for better inference and thus hopefully better fit outside of it. That's what shrinkage does. Of course, your model will never perfectly retrodict the data, but now we should expect a systematic difference.

Second, "preduction" in a multilevel model requires choices. If we want to validate it against the clusters used to train the model, that's one thing. But if we want instead to predict new clusters, that is something else.

### 4.1 Posterior predictions for the same clusters

When you're working with the same clusters used to train the data, the varying intercepts are just paaters. The only trick is to ensure that you use the right intercept for each class of the data. If you're using `link` and `sim` to do this, then it's handled for you; otherwise you need to use the model definition. We'll do it both ways.

Once again, because the partial pooling 'shrinks' the data toward the mean, we shouldn't expect the posterior distribution to match the raw data.


```R
chimp <- 2
d_pred <- list(
    actor = rep(chimp, 4),
    treatment = 1:4,
    block_id = rep(1, 4)
)

p <- link(m13.4, data = d_pred)
p_mu <- apply(p, 2, mean)
p_ci <- apply(p, 2, PI)
```

We can also do this directly from the samples. The only trick is that when we work with samples from the posterior, the varying intercepts will be a matrix of samples.


```R
post <- extract.samples(m13.4)
str(post)
```

    List of 7
     $ b      : num [1:2000, 1:4] 0.0605 0.0384 0.2274 -0.012 0.1415 ...
     $ a      : num [1:2000, 1:7] -0.734 -0.593 -1.011 -0.755 -0.594 ...
     $ g      : num [1:2000, 1:6] -0.2779 -0.3638 0.0514 -0.2128 -0.0297 ...
     $ a_bar  : num [1:2000, 1] 0.472 0.828 -0.902 0.18 0.207 ...
     $ sigma_a: num [1:2000, 1] 2.97 3.07 1.9 1.77 1.56 ...
     $ sigma_g: num [1:2000, 1] 0.286 0.41 0.197 0.105 0.149 ...
     $ p      : num [1:2000, 1:504] 0.279 0.285 0.325 0.273 0.382 ...
     - attr(*, "source")= chr "ulam posterior from object"



```R
# plotting the density for actor 5
par(bg = 'white')
dens(post$a[, 5])
```


    
![png](chapter_13_images/output_62_0.png)
    


To construct the posterior predictions, we build our own link function.


```R
p_link <- function(treatment, actor = 1, block_id = 1) {
    logodds <- with(post,
        a[, actor] + g[, block_id] + b[, treatment]
    )
    return(inv_logit(logodds))
}
```


```R
p_raw <- sapply(1:4, function(i) p_link(i, actor = 2, block_id = 1))
p_mu <- apply(p_raw, 2, mean)
p_ci <- apply(p_raw, 2, PI)
```

### 4.2 Posterior prediction for new clusters

When we're making predictions for new clusters, we're trying to generalize the model to go outside of our data. There's no one, cut-and-dried approach to this; it will depend on the situation, the data, the model, &c.

As an example, let's try to imagine that we are running the chimpanzee experiment with new chimpanzees. ATM we have a bunch of actor-level intercepts for the ones in the trial, but that's not really helpful because we are now trying to make the model work on new actors.

However, we can make use of the $\bar{\alpha}$ and $\sigma_\alpha$ parameters (the adaptive prior parameters)!

One approach might be to make the prediction for an 'average' actor (one whose mean is $\bar{\alpha}$).

We'll have to make a new link function:


```R
p_link_abar <- function(treatment) {
    logodds <- with(post,
        a_bar + b[, treatment]
    )
    return(inv_logit(logodds))
}
```

(We're ignoring `block` because this trial will be using new blocks and we're assuming that the effect of the block is basically zero anyway).


```R
post <- extract.samples(m13.4)
p_raw <- sapply(1:4, function(i) p_link_abar(i))
p_mu <- apply(p_raw, 2, mean)
p_ci <- apply(p_raw, 2, PI)

plot_df <- data.frame(
    treatment=c("R/N", "L/N", "R/P", "L/P"),
    mean = p_mu,
    lower = p_ci[1, ],
    upper = p_ci[2, ]
)
plot_df$treatment <- factor(plot_df$treatment, levels = plot_df$treatment)
# plot_df
ggplot(plot_df, aes(treatment, group = 1)) +
    geom_line(aes(y = mean)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2)
```


    
![png](chapter_13_images/output_69_0.png)
    


This is great, but it doesn't really show the potential variation among the new actors. This is strictly for an average chimpanzee!

To capture the variation, let's simulate a bunch of new chimpanzees using `rnorm` with mean `a_bar` and sd `sigma_a`! We'll have to do this outside of the link function because we want to be referencing the same population of simulated chimpanzees regardless of the treatment.


```R
a_sim <- with(post,
    rnorm(length(post$a_bar), a_bar, sigma_a)
)
p_link_asim <- function(treatment) {
    logodds <- with(post,
        a_sim + b[, treatment]
    )
    return(inv_logit(logodds))
}
p_raw_asim <- sapply(1:4, function(treatment) p_link_asim(treatment))
```


```R
p_mu <- apply(p_raw_asim, 2, mean)
p_ci <- apply(p_raw_asim, 2, PI)

plot_df <- data.frame(
    treatment=c("R/N", "L/N", "R/P", "L/P"),
    mean = p_mu,
    lower = p_ci[1, ],
    upper = p_ci[2, ]
)
plot_df$treatment <- factor(plot_df$treatment, levels = plot_df$treatment)
# plot_df
ggplot(plot_df, aes(treatment, group = 1)) +
    geom_line(aes(y = mean)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
    labs(title = "Marginal of Actors")
```


    
![png](chapter_13_images/output_72_0.png)
    


This is obviously showing something different. Instead of the expected results from a perfectly average actor, this is the result of a bunch of roughly average actors with the kind of variation that we saw in the data.

So... which of these approaches (single average actor or roughly average population) should we use? It depends on the question being asked! The single average actor is more useful when trying to see the effect of the treatment, whereas the population results are more useful in terms of letting us know what we can expect from our next experiment.

In fact, we can do better - we can simply plot the results for each of the simulated actors as discrete lines!


```R
plot_df <- data.frame(
    treatment= character(),
    p = numeric(),
    actor = integer()
)

for (i in 1:100) {
    plot_df <- rbind(plot_df,
        data.frame(
            treatment = c("R/N", "L/N", "R/P", "L/P"),
            p = p_raw_asim[i, ],
            actor = i
        )
    )
}
plot_df$treatment <- factor(plot_df$treatment, levels = c("R/N", "L/N", "R/P", "L/P"))

ggplot(plot_df, aes(treatment, p, group = actor)) +
    geom_line()
```


    
![png](chapter_13_images/output_74_0.png)
    


One interesting thing to note is that for actors with very large or very small intercepts (near the top or bottom of the plot), the treatment has esically no effect. This is because they start with a very strong preference for pulling right or left! For the ones in the middle (no strong 'handedness') we see a big effect.

## 5 Post-stratification

A common problem is how to take a non-representative sample and extrapolate to the rest of the population. For instance, you might ask a bunch of voters how they're going to vote. However, some groups will respond at different rates, and so you need to account for this. The survey is biased by the response rate.

On technique is [[Post-Stratification]]. The idea is to fit a model where each slice of the population (age, economic, education, &c.) has its own voting intention. Then the estimates of these intentions are re-weighted using a census of the full population. However, some of the groups can be small, with only a few estimates. Thus, post-stratification is often combined with multilevel modelling, in which case with is called [[Multilevel Modelling and Post-Stratification]], or MRP (pronounced "Mister P").

How does this work?

Say you have estimates $p_i$ for each category. Then the post-stratified prediction for the entire population is a re-weighting using the number of people in each category:

$$
\frac{\sum_i N_i p_i}{\sum_i N_i}
$$

Post-stratification doesn't always work. For instance, when selection bias is caused by the outcome of interest. For instance, suppose that responding to the survey, $R$, is influenced by age $A$, and that age $A$ influences boting entention $V$: $R \leftarrow A \to V$. The it is possible to estimate the influence of $A$ on $V$. But if $V \to R$, then there is little hope. Suppose that only supporters responded. Then $V=1$ for everyone who responds, which will of course make generalization to the rest of the population impossible. Selection on the outcome variable is one of the worst things that can happen in statistics.

A general framework for generalizability is [[Transportability]]. Post-stratification is a special case of this framework, as are meta-analyses and the application of estimates across populations.

## 6 Practice

**13E1** Which of the following priors will produce more shinkage in the estimate? $\alpha_[\text{TANK}] \sim \text{Normal}(0, 1)$ or $\alpha_[\text{TANK}] \sim \text{Normal}(0, 2)$

**Answer** The first one will. It's a stronger prior, and will thus tend to cause the other estimates to 'shrink' toward it more.

**13E2** Rewrite the following as a multilevel model

$$
\begin{align*}
y_i &\sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) &= \alpha_{ \text{GROUP}[i]  } + \beta x_i \\
\alpha_{\text{GROUP}} &\sim \text{Normal(0, 1.5)} \\
\beta &\sim \text{Normal}(0, 0.5) \\
\end{align*}
$$

**Answer** Basically, we want to have an adaptive prior here. What we'll do is to make the prior for $\alpha$ based on the group level

$$
\begin{align*}
y_i &\sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) &= \alpha_{ \text{GROUP}[i]  } + \beta x_i \\
\alpha_{\text{GROUP}} &\sim \text{Normal}(\bar{\alpha}, \sigma_\alpha) \\
\bar\alpha &\sim \text{Normal(0, 1)} \\
\sigma_\alpha &\sim \text{Exponential}(1) \\
\beta &\sim \text{Normal}(0, 0.5) \\
\end{align*}
$$

(The numbers for the new hyperpriors, $\bar\alpha$ and $\sigma_\alpha$, and largely made up; without knowing more about the situation it's hard to evaluate how accurate they are).

**13E3** Rewrite the following as a multilevel model

$$
\begin{align*}
y_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha_{\text{GROUP}[i]} + \beta x_i \\
\alpha_{\text{GROUP}} &\sim \text{Normal}(0, 5) \\
\beta &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$

**Answer** Same as before, we'll replace the prior for $\alpha$ with an adaptive one:

$$
\begin{align*}
y_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha_{\text{GROUP}[i]} + \beta x_i \\
\alpha_{\text{GROUP}} &\sim \text{Normal}(\bar\alpha, \sigma_\alpha) \\
\bar\alpha &\sim \text{Normal}(0, 1) \\
\sigma_\alpha &\sim \text{Exponential}(2) \\
\beta &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$

**13E4** Write a mathemtical model formulat for a Poisson regression with varying intercepts.

**Answer** For this, we're going to take a normal Poisson regression and include adaptive parameters (the varying intercepts):
$$
\begin{align*}
y_i &\sim \text{Poisson}(\lambda_i) \\
\log \lambda_i &\sim \alpha_{ \text{GROUP}[i] } \\
\alpha_\text{GROUP} &\sim \text{Normal}(\bar\alpha, \sigma_\alpha) \\
\bar\alpha &\sim \text{Normal}(0, 1) \\
\sigma_\alpha &\sim \text{Exponential}(1)
\end{align*}
$$

**13E5** Write a mathematical model formula for a Poisson regression with two different kinds of varying intercepts, a cross-classified model.

**Answer** Here we'll introduce another intercept ($\beta$) which tracks a made-up BLOCK (so now each item is part of a GROUP and a BLOCK):
$$
\begin{align*}
y_i &\sim \text{Poisson}(\lambda_i) \\
\log \lambda_i &\sim \alpha_{\text{GROUP}[i]} + \beta_{\text{BLOCK}[i]} \\
\alpha_\text{GROUP} &\sim \text{Normal}(\bar\alpha, \sigma_\alpha) \\
\beta_\text{BLOCK} &\sim \text{Normal}(\bar\beta, \sigma_\beta) \\
\bar\alpha &\sim \text{Normal}(0, 1) \\
\sigma_\alpha &\sim \text{Exponential}(1) \\
\bar\beta &\sim \text{Normal}(0, 1) \\
\sigma_\beta &\sim \text{Exponential}(1)
\end{align*}
$$

**13M1** Revisit the Reed frog survival data, `data(reedfrogs)`, and add the `predation` and `size` treatment variables to the varying intercepts model. Consider models with either main effect along,both main effects, as well as a model including both and their interaction. Instead of focusing on inferences about thoses two predictor variables, focus on the inferred variation across tanks. Explain why it changes as it does across models.

**Answer**


```R
data(reedfrogs)
d <- reedfrogs
str(d)
```

    'data.frame':	48 obs. of  5 variables:
     $ density : int  10 10 10 10 10 10 10 10 10 10 ...
     $ pred    : Factor w/ 2 levels "no","pred": 1 1 1 1 1 1 1 1 2 2 ...
     $ size    : Factor w/ 2 levels "big","small": 1 1 1 1 2 2 2 2 1 1 ...
     $ surv    : int  9 10 7 10 9 9 10 9 4 9 ...
     $ propsurv: num  0.9 1 0.7 1 0.9 0.9 1 0.9 0.4 0.9 ...


Recall that the varying effects model was

$$
\begin{align*}
S_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma) & \text{for $j = 1\dots 48$} \\
\bar{\alpha} &\sim \text{Normal}(0, 1.5) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$


```R
d$tank <- 1:nrow(d)

dat <- list(
    S = d$surv,
    N = d$density,
    tank = d$tank
)
m13m1.1 <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.5 seconds.

Now we need

a) predation only
b) size only
c) predation and size
d) predation, size, and their interaction

For both of these, the variables are factors $\to$ we need to switch them for variables / indices:


```R
d$pred_index <- ifelse(d$pred == 'no', 1L, 2L)
d$size_index <- ifelse(d$size == 'small', 1L, 2L)
d$pred_size_index <- 2 * (d$pred_index - 1) + (d$size_index - 1) + 1

summary(d)
```


        density        pred       size         surv          propsurv     
     Min.   :10.00   no  :24   big  :24   Min.   : 4.00   Min.   :0.1143  
     1st Qu.:10.00   pred:24   small:24   1st Qu.: 9.00   1st Qu.:0.4964  
     Median :25.00                        Median :12.50   Median :0.8857  
     Mean   :23.33                        Mean   :16.31   Mean   :0.7216  
     3rd Qu.:35.00                        3rd Qu.:23.00   3rd Qu.:0.9200  
     Max.   :35.00                        Max.   :35.00   Max.   :1.0000  
          tank         pred_index    size_index  pred_size_index
     Min.   : 1.00   Min.   :1.0   Min.   :1.0   Min.   :1.00   
     1st Qu.:12.75   1st Qu.:1.0   1st Qu.:1.0   1st Qu.:1.75   
     Median :24.50   Median :1.5   Median :1.5   Median :2.50   
     Mean   :24.50   Mean   :1.5   Mean   :1.5   Mean   :2.50   
     3rd Qu.:36.25   3rd Qu.:2.0   3rd Qu.:2.0   3rd Qu.:3.25   
     Max.   :48.00   Max.   :2.0   Max.   :2.0   Max.   :4.00   


So then our model for e.g. including predation look like

$$
\begin{align*}
S_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} + \beta_{\text{PREDATION}[i]} \\
\beta &\sim \text{Normal}(0, 1.5) \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma) & \text{for $j = 1\dots 48$} \\
\bar{\alpha} &\sim \text{Normal}(0, 1.5) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$

size only looks like
$$
\begin{align*}
S_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} + \gamma_{\text{SIZE}[i]} \\
\gamma &\sim \text{Normal}(0, 1.5) \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma) & \text{for $j = 1\dots 48$} \\
\bar{\alpha} &\sim \text{Normal}(0, 1.5) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$

both (no interaction) looks like

$$
\begin{align*}
S_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} + \beta_{\text{PREDATION}[i]} + \gamma_{\text{SIZE}[i]} \\
\beta &\sim \text{Normal}(0, 1.5) \\
\gamma &\sim \text{Normal}(0, 1.5) \\
\eta &\sim \text{Normal}(0, 1.5) \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma) & \text{for $j = 1\dots 48$} \\
\bar{\alpha} &\sim \text{Normal}(0, 1.5) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$

and both (with interaction) looks like

$$
\begin{align*}
S_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{TANK}[i]} + \beta_{\text{PREDATION}[i]} + \gamma_{\text{SIZE}[i]} + \eta_{\text{PREDATION\_SIZE}[i]}\\
\beta &\sim \text{Normal}(0, 1.5) \\
\gamma &\sim \text{Normal}(0, 1.5) \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma) & \text{for $j = 1\dots 48$} \\
\bar{\alpha} &\sim \text{Normal}(0, 1.5) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$


```R
dat <- list(
    S = d$surv,
    N = d$density,
    tank = d$tank,
    pred_index = d$pred_index,
    size_index = d$size_index,
    pred_size_index = d$pred_size_index
)
# predation
m13m1.pred <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + b[pred_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        b[pred_index] ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)

# size
m13m1.size <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + g[size_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        g[size_index] ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)

# predation and size (no interaction)
m13m1.pred_size_no_interaction <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + b[pred_index] + g[size_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        b[pred_index] ~ dnorm(0, 1.5),
        g[size_index] ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)

```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.2 seconds.
    Total execution time: 1.3 seconds.

```R
# predation and size (interaction)
m13m1.pred_size_interaction <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + b[pred_index] + g[size_index] + e[pred_size_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        b[pred_index] ~ dnorm(0, 1.5),
        g[size_index] ~ dnorm(0, 1.5),
        e[pred_size_index] ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.4 seconds.
    Total execution time: 2.2 seconds.

```R
par(bg = 'white')
plot(precis(m13m1.pred, depth = 2), main = "Predation Only")
```

    
![png](chapter_13_images/output_85_0.png)

```R
par(bg = 'white')
plot(precis(m13m1.size, depth = 2), main = "Size Only")
```
    
![png](chapter_13_images/output_86_0.png)
    

```R
par(bg = 'white')
plot(precis(m13m1.pred_size_no_interaction, depth = 2), main = "Predation and Size, no Interaction")
```

![png](chapter_13_images/output_87_0.png)
    



```R
par(bg = 'white')
plot(precis(m13m1.pred_size_interaction, depth = 2), main = "Predation and Size, with Interaction")
```


    
![png](chapter_13_images/output_88_0.png)
    


If we're looking at the variation across tanks, that's encoded in the `sigma` parameter. Let's look at that.


```R
pred_only <- extract.samples(m13m1.pred)$sigma
size_only <- extract.samples(m13m1.size)$sigma
pred_size_no_interaction <- extract.samples(m13m1.pred_size_no_interaction)$sigma
pred_size_interaction <- extract.samples(m13m1.pred_size_interaction)$sigma

model_samples <- list(pred_only, size_only, pred_size_no_interaction, pred_size_interaction)
```


```R
model_descriptions <- c("Predation Only", "Size Only", "Predation & Size (No Interaction)", "Predation & Size (Interaction)")
plot_df <- data.frame(
    model = factor(model_descriptions, levels = model_descriptions),
    mean = sapply(model_samples, mean),
    lower = sapply(model_samples, function(c) quantile(c, 0.025)),
    upper = sapply(model_samples, function(c) quantile(c, 0.975))
)
plot_df


ggplot(plot_df, aes(model)) +
    geom_pointrange(aes(y = mean, ymin = lower, ymax = upper))
```


<table class="dataframe">
<caption>A data.frame: 4 × 4</caption>
<thead>
	<tr><th scope=col>model</th><th scope=col>mean</th><th scope=col>lower</th><th scope=col>upper</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Predation Only                   </td><td>0.8172831</td><td>0.5523644</td><td>1.119904</td></tr>
	<tr><td>Size Only                        </td><td>1.6137932</td><td>1.2608868</td><td>2.097616</td></tr>
	<tr><td>Predation &amp; Size (No Interaction)</td><td>0.7764043</td><td>0.5063266</td><td>1.110379</td></tr>
	<tr><td><span style=white-space:pre-wrap>Predation &amp; Size (Interaction)   </span></td><td>0.7597950</td><td>0.5167539</td><td>1.069664</td></tr>
</tbody>
</table>




    
![png](chapter_13_images/output_91_1.png)
    


From the above, it's clear that `sigma` is materially larger for the size only model and basically the same for each of the other models.

Looking at the earlier plots of the parameter values, it seems that what is happening is that the absence or presence of predation is having a large explanatory effect - the $\beta$ parameters are large. Since they have such a good explanatory effect, the model that's missing them, the "Size Only" model, has incorporated the uncertainty into a larger $\sigma$ variable.

**13M2** Compare the models you fit using WAIC. Can you reconcile the differences in WAIC with the posterior distributions of the model?

**Answer**


```R
compare(m13m1.pred, m13m1.size, m13m1.pred_size_no_interaction, m13m1.pred_size_interaction, func = WAIC)
```


<table class="dataframe">
<caption>A compareIC: 4 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m13m1.pred</th><td>199.2979</td><td>9.317249</td><td>0.000000</td><td>      NA</td><td>19.22281</td><td>0.4146400</td></tr>
	<tr><th scope=row>m13m1.pred_size_interaction</th><td>200.6637</td><td>9.650396</td><td>1.365797</td><td>3.338249</td><td>19.71828</td><td>0.2094557</td></tr>
	<tr><th scope=row>m13m1.pred_size_no_interaction</th><td>200.7702</td><td>8.893776</td><td>1.472390</td><td>2.181174</td><td>19.45172</td><td>0.1985847</td></tr>
	<tr><th scope=row>m13m1.size</th><td>200.9968</td><td>7.297595</td><td>1.698914</td><td>6.351048</td><td>21.23925</td><td>0.1773196</td></tr>
</tbody>
</table>



From the above, it seems that the predation only model does the best job of explaining our results. Basically, it is telling us that the main driver of mortality differences is predation, not the tank size.

**13M3** Re-estimate the basic Reed frog varying intercept model, but now using a Cauchy distribution in palce of the Gaussian distribution for the varying intercepts.

$$
\begin{align*}
    s_i &\sim \text{Binomial}(n_i, p_i) \\
    \text{logit}(p_i) &= \alpha_{\text{TANK}[i]} \\
    \alpha_{\text{TANK}} &\sim \text{Cauchy}(\bar{\alpha}, \sigma) \\
    \bar\alpha &\sim \text{Normal}(0, 1) \\
    \sigma &\sim \text{Exponential}(1)
\end{align*}
$$

(You are likely to see many divergent transitions for this model. Can you figure out why? Can you fix them?) Compare the posterior means of the intercepts, $\alpha_{\text{TANK}}$, to the posterior means produced in the chapter, using the customary Gaussian prior. Can you explain the pattern of differences? Take note of any change in the mean $\alpha$ as well.

**Answer**

First, let's naively just try to run the model.


```R
m13m1.1.cauchy.1 <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank],
        a[tank] ~ dcauchy(a_bar, sigma),
        a_bar ~ dnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.8 seconds.
    Total execution time: 3.5 seconds.
	
    Warning: 11 of 2000 (1.0%) transitions hit the maximum treedepth limit of 10.
    See https://mc-stan.org/misc/warnings for details.


```R
par(bg = 'white')
plot(precis(m13m1.1.cauchy.1, depth = 2))
```


    
![png](chapter_13_images/output_98_0.png)
    


Wow, that's a lot of divergent transitions! Probably this is happening because the Cauchy distribution has very wide tails, which means that the parameters are free to wander all over the potential space. The first thing we can try is to change the `adapt_delta` parameter:


```R
m13m1.1.cauchy.2 <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank],
        a[tank] ~ dcauchy(a_bar, sigma),
        a_bar ~ dnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE,
    control = list(adapt_delta = 0.99)
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.9 seconds.
    Total execution time: 3.9 seconds.

    Warning: 442 of 2000 (22.0%) transitions hit the maximum treedepth limit of 10.
    See https://mc-stan.org/misc/warnings for details.

Hmm, well, that didn't appear to have helped. Looks like we'll have to try something different! Let's see if centring the model helps.

```R
par(bg = 'white')
plot(precis(m13m1.1.cauchy.2, depth = 2))
```

    
![png](chapter_13_images/output_102_0.png)
    

```R
m13m1.1.cauchy.3 <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a_bar + sigma * z[tank],
        # a[tank] ~ dcauchy(a_bar, sigma),
        # a[tank] <- a_bar + sigma * z[tank],
        z[tank] ~ dcauchy(0, 1),
        a_bar ~ dnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE,
    # control = list(adapt_delta = 0.99)
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.6 seconds.
    Total execution time: 2.6 seconds.
    
    Warning: 1 of 4 chains had an E-BFMI less than 0.3.
    See https://mc-stan.org/misc/warnings for details.
    
So uncentring the model definitely worked!


```R
base_model_samples <- extract.samples(m13m1.1)
centred_cauchy_samples <- extract.samples(m13m1.1.cauchy.3)
```


```R
# NB we don't have an a[tank] parameter in the cauchy model, so we have to recreate it. The variable a[tank] would be logit(p), so that's what we'll do from the samples
cauchy_a_tank <- logit(centred_cauchy_samples$p)
head(cauchy_a_tank)
```


<table class="dataframe">
<caption>A matrix: 6 × 48 of type dbl</caption>
<tbody>
	<tr><td>1.4066056</td><td>2.458186</td><td>0.8764016</td><td>     Inf</td><td>3.2268151</td><td>1.268968</td><td>1.571561</td><td>0.8834904</td><td> 0.4681593</td><td>0.8121948</td><td>⋯</td><td>2.762528</td><td>1.892383</td><td>-1.730036</td><td>-0.64013953</td><td>-1.23210622</td><td>-0.4764193</td><td>0.074446364</td><td>-0.7768416</td><td>1.8328930</td><td> 0.42972082</td></tr>
	<tr><td>2.0727775</td><td>2.511556</td><td>1.2679989</td><td>     Inf</td><td>0.6063906</td><td>1.873060</td><td>4.290573</td><td>2.3988950</td><td>-0.9724170</td><td>2.1820283</td><td>⋯</td><td>2.150056</td><td>1.831901</td><td>-2.027906</td><td>-0.49260962</td><td> 0.27643092</td><td>-0.1609747</td><td>1.389841886</td><td>-0.4604841</td><td>1.6648127</td><td>-0.53784084</td></tr>
	<tr><td>1.3355307</td><td>2.391737</td><td>0.6285958</td><td>10.92512</td><td>3.1082068</td><td>1.696171</td><td>2.359901</td><td>1.0594425</td><td> 0.4820295</td><td>1.4652019</td><td>⋯</td><td>2.623741</td><td>3.130563</td><td>-2.408541</td><td>-1.24798004</td><td>-0.81506136</td><td>-0.6997259</td><td>0.004584008</td><td>-0.2528872</td><td>2.2798807</td><td> 0.31440867</td></tr>
	<tr><td>1.3246783</td><td>1.919243</td><td>0.4364256</td><td>13.12236</td><td>1.1812807</td><td>1.582378</td><td>1.678109</td><td>2.0738859</td><td> 0.1497472</td><td>2.3052147</td><td>⋯</td><td>1.937298</td><td>2.949228</td><td>-1.699742</td><td>-0.04210222</td><td>-0.05707549</td><td> 0.5146120</td><td>1.186792976</td><td>-0.8367439</td><td>1.5729799</td><td> 0.06392576</td></tr>
	<tr><td>1.1318763</td><td>2.131613</td><td>1.7582871</td><td>12.42921</td><td>1.4232154</td><td>1.651051</td><td>2.357266</td><td>2.6210327</td><td>-0.8336256</td><td>1.5030572</td><td>⋯</td><td>2.817308</td><td>2.516103</td><td>-2.505457</td><td>-0.45573115</td><td>-0.40864946</td><td>-0.6518889</td><td>0.276137389</td><td>-0.1249182</td><td>1.3351850</td><td> 0.21483044</td></tr>
	<tr><td>0.9496212</td><td>4.768450</td><td>1.8790416</td><td>11.41760</td><td>1.6374011</td><td>1.302962</td><td>2.600908</td><td>1.6895198</td><td>-0.2137704</td><td>0.9727636</td><td>⋯</td><td>2.721475</td><td>2.767804</td><td>-2.311445</td><td>-0.20112323</td><td>-0.07801555</td><td>-0.3091223</td><td>0.281239183</td><td>-0.8319827</td><td>0.9338958</td><td> 0.33249367</td></tr>
</tbody>
</table>




```R
base_means <- apply(base_model_samples$a, 2, mean)
base_ci <- apply(base_model_samples$a, 2, PI)
cauchy_means <- apply(cauchy_a_tank, 2, mean)
cauchy_ci <- apply(cauchy_a_tank, 2, PI)

plot_df <- data.frame(
    tank = rep(d$tank, 2),
    mean = c(base_means, cauchy_means),
    lower = c(base_ci[1, ], cauchy_ci[1, ]),
    upper = c(base_ci[2, ], cauchy_ci[2, ]),
    model = c(rep("Gaussian", length(d$tank)), rep("Cauchy", length(d$tank)))
)
ggplot(plot_df, aes(tank, group = model, colour = model)) +
    geom_pointrange(aes(y = mean, ymin = lower, ymax = upper), position = position_dodge(width = 0.5))
```


    
![png](chapter_13_images/output_107_0.png)
    


These look basically the same, except that when the value for `a[tank]` is large in the Gaussian model it tends to be even larger in the Cauchy one. Let's plot these against each other to see if that makes the relationship clearer.

Now let's take a look at $\bar\alpha$!


```R
plot_df <- data.frame(
    gaussian_a = base_means,
    cauchy_a = cauchy_means,
    tank = d$tank
)
ggplot(plot_df, aes(gaussian_a, cauchy_a)) +
    geom_point() +
    coord_cartesian(ylim = c(-2.5, 10))
```


    
![png](chapter_13_images/output_109_0.png)
    


Now we're running into a problem - some of the Cauchy values are infinity!


```R
cauchy_means
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1.99097607132162</li><li>Inf</li><li>1.0884630394608</li><li>Inf</li><li>2.00674867313877</li><li>2.00807848721817</li><li>Inf</li><li>2.02505695144336</li><li>-0.0591642435338424</li><li>1.95440720336725</li><li>1.09641849208892</li><li>0.746650796203728</li><li>1.09571548538488</li><li>0.375871655518004</li><li>1.98621139423836</li><li>1.9936691189107</li><li>2.84477472795822</li><li>2.24731053086973</li><li>1.89766221902871</li><li>Inf</li><li>2.27468866344778</li><li>2.26947567725765</li><li>2.26233015768069</li><li>1.66097267885665</li><li>-1.06268157844437</li><li>0.236228807470768</li><li>-1.56893246337781</li><li>-0.441864986736109</li><li>0.236955749404133</li><li>1.42329918077624</li><li>-0.638120949732484</li><li>-0.29033170361516</li><li>3.28918249122887</li><li>2.63984123876756</li><li>2.5996004685938</li><li>1.97672851495113</li><li>1.97845848105076</li><li>Inf</li><li>2.59102742177867</li><li>2.2544864288904</li><li>-2.0068652130617</li><li>-0.563133424348994</li><li>-0.431570439200762</li><li>-0.318292989592934</li><li>0.637390128541772</li><li>-0.569943881827886</li><li>1.95278843877711</li><li>0.0424702410922158</li></ol>

Probably what is happening here is that the hierarchical priors are 'shrinking' the Gaussian models more toward the mean (since the extreme `a[tank]` values are ones where a high proportion of the tadpoles survived), leading to very large (infinite) values in the Cauchy model.

Now let's compare the posterior means of the intercepts ($\alpha_{\text{TANK}}$)


```R
plot_df <- data.frame(
    mean = c(mean(base_model_samples$a_bar), mean(centred_cauchy_samples$a_bar)),
    lower = c(PI(base_model_samples$a_bar)[1], PI(centred_cauchy_samples$a_bar)[1]),
    upper = c(PI(base_model_samples$a_bar)[2], PI(centred_cauchy_samples$a_bar)[2]),
    model = c("Gaussian", "Cauchy")
)
ggplot(plot_df, aes(model)) +
    geom_pointrange(aes(y = mean, ymin = lower, ymax = upper))
```


    
![png](chapter_13_images/output_113_0.png)
    


These look basically identical! So it looks like the model only affected the `a[tank]` values.

**13M4** Now use the Student-t distribution with $\nu = 2$ for the intercepts:

$$
\alpha_{\text{TANK}} \sim \text{Student}(2, \alpha, \sigma)
$$

Compare the resulting posterior to both the original model and the Cauchy model in **13M3**. Can you explain the differences and similarities in shinkage in terms of the properties of these distributions?

**Answer**


```R
m13m3.student <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank],
        a[tank] ~ dstudent(2, a_bar, sigma),
        a_bar ~ dnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE,
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.2 seconds.
    Total execution time: 0.9 seconds.

```R
# campare the a_tank between the different models

student_model_samples <- extract.samples(m13m3.student)

base_means <- apply(base_model_samples$a, 2, mean)
base_ci <- apply(base_model_samples$a, 2, PI)
cauchy_means <- apply(cauchy_a_tank, 2, mean)
cauchy_ci <- apply(cauchy_a_tank, 2, PI)
student_means <- apply(student_model_samples$a, 2, mean)
student_ci <- apply(student_model_samples$a, 2, PI)

student_a_bar <- apply(student_model_samples$a_bar, 2, mean)

plot_df <- data.frame(
    tank = rep(d$tank, 3),
    mean = c(base_means, cauchy_means, student_means),
    lower = c(base_ci[1, ], cauchy_ci[1, ], student_ci[1, ]),
    upper = c(base_ci[2, ], cauchy_ci[2, ], student_ci[2, ]),
    model = rep(c("Gaussian", "Cauchy", "Student-t"), each = length(d$tank))
)
ggplot(plot_df, aes(tank, group = model, colour = model)) +
    geom_pointrange(aes(y = mean, ymin = lower, ymax = upper), position = position_dodge(width = 0.5)) +
    geom_hline(aes(yintercept = student_a_bar), linetype = 'dashed') +
    annotate('label', x = 45, y = student_a_bar + 0.5, label = "bar(alpha)", parse = TRUE)
```


    
![png](chapter_13_images/output_117_0.png)
    


From the graph, the Student-t distribution experienced an amount of shrinkage intermediate between the Gaussian and the Cauchy. One thing to note is that the colours have changed compared to the earlier graph.

This is as expected. A more concentrated prior will tend to produce more shrinkage toward the mean. Of the three distributions, the Gaussian is the most concentrated, followed by the Student-t and then the Cauchy distribution. This fact explains the different levels of shrinkage.

One thing to note: Normal = $\text{Student-t}(\nu = \infty)$, and Cauchy = $\text{Student-t}(\nu = 1)$, so it makes sense to use a Student-t and then muck about with the number of degrees of freedom to achieve the level of shrinkage that you're looking for.

**13M5** Modify the cross-classified chimpanzee model `m13.4` so that the adaptive prior for blocks contains a parameter $\bar\gamma$ for its mean:

$$
\begin{align*}
\gamma_j &\sim \text{Normal}(\bar\gamma, \sigma_\gamma) \\
\bar\gamma &\sim \text{Normal}(0, 1.5)
\end{align*}
$$

Compare this model to `m13.4`. What has including $\bar\gamma$ done?

**Answer**

Original model:
$$
\begin{align*}
L_i &\sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{ACTOR}[i]} + \gamma_{\text{BLOCK}[i]} + \beta_{\text{TREATMENT}[i]} \\
\beta_j &\sim \text{Normal}(0, 0.5) & \text{for $j = 1\dots 4$} \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma_{\alpha}) & \text{for $j = 1\dots 7$} \\
\gamma_j &\sim \text{Normal}(0, \sigma_{\gamma}) & \text{for $j = 1\dots 6$} \\
\bar{\alpha} &\sim \text{Normal(0, 1.5)} \\
\sigma_\alpha &\sim \text{Exponential(1)} \\
\sigma_\gamma &\sim \text{Exponential(1)} \\
\end{align*}
$$

New model:
$$
\begin{align*}
L_i &\sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{ACTOR}[i]} + \gamma_{\text{BLOCK}[i]} + \beta_{\text{TREATMENT}[i]} \\
\beta_j &\sim \text{Normal}(0, 0.5) & \text{for $j = 1\dots 4$} \\
\alpha_j &\sim \text{Normal}(\bar{\alpha}, \sigma_{\alpha}) & \text{for $j = 1\dots 7$} \\
\gamma_j &\sim \text{Normal}(\bar\gamma, \sigma_{\gamma}) & \text{for $j = 1\dots 6$} \\
\bar{\alpha} &\sim \text{Normal(0, 1.5)} \\
\bar{\gamma} &\sim \text{Normal}(0, 1.5) \\
\sigma_\alpha &\sim \text{Exponential(1)} \\
\sigma_\gamma &\sim \text{Exponential(1)} \\
\end{align*}
$$


```R
data(chimpanzees)
d <- chimpanzees
d$treatment <- 1 + d$prosoc_left + 2 * d$condition

dat_list <- list(
    pulled_left = d$pulled_left,
    actor = d$actor,
    block_id = d$block,
    treatment = as.integer(d$treatment)
)

set.seed(13)
m13m5 <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + g[block_id] + b[treatment],
        b[treatment] ~ dnorm(0, 0.5),
        # adaptive priors
        a[actor] ~ dnorm(a_bar, sigma_a),
        g[block_id] ~ dnorm(sigma_bar, sigma_g),
        # hyper priors
        a_bar ~ dnorm(0, 1.5),
        sigma_bar ~ dnorm(0, 1.5),
        sigma_a ~ dexp(1),
        sigma_g ~ dexp(1)
    ),
    data = dat_list,
    chains = 4,
    cores = 4,
    log_lik = TRUE,
    control = list(adapt_delta = 0.99)
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 8.5 seconds.
    Total execution time: 14.0 seconds.

    Warning: 35 of 2000 (2.0%) transitions ended with a divergence.
    See https://mc-stan.org/misc/warnings for details.
    
    Warning: 290 of 2000 (14.0%) transitions hit the maximum treedepth limit of 10.
    See https://mc-stan.org/misc/warnings for details.
    
```R
par(bg = 'white')
plot(precis(m13.4, depth = 2))
plot(precis(m13m5, depth = 2))
```


    
![png](chapter_13_images/output_120_0.png)
    



    
![png](chapter_13_images/output_120_1.png)
    



```R
compare(m13.4, m13m5)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m13.4</th><td>532.0367</td><td>19.39689</td><td>0.000000</td><td>      NA</td><td>10.59352</td><td>0.5965073</td></tr>
	<tr><th scope=row>m13m5</th><td>532.8186</td><td>19.38993</td><td>0.781866</td><td>0.273291</td><td>11.02560</td><td>0.4034927</td></tr>
</tbody>
</table>



So this didn't really help! All of the parameters of the second model (with the other adaptive prior). My suspicion is that we're running into the same problem that we did earlier with multicollinearity (although that's probably the wrong term). Basically, we have two adative priors with two different means; when we add them together, there are lots of different values for each of the means that result in the same sum. As a result, the level of confidence in each one goes down (wider variance).

This seems like a model specification issue - it's probably not a good idea to have multiple adaptive priors in the same sum!

**13M6** Sometimes prior and data (through the likelihood) are in conflict because they concentrate around different regions of the parameter space. What happens in these cases depends a lot upon the shape of the tails of the disitrubion. Likewisde, the tails of distributions strongly influence how outliers are shrunk or not towards the mean. Consider four different models to fit to one observation at $y = 0$. The models differ only in the distributions assigned to the likelihood and prior.

Model NN:
$$
\begin{align*}
y &\sim \text{Normal}(\mu, 1) \\
\mu &\sim \text{Normal}(10, 1) \\
\end{align*}
$$

Model TN:
$$
\begin{align*}
y &\sim \text{Student}(2, \mu, 1) \\
\mu &\sim \text{Normal}(10, 1) \\
\end{align*}
$$

Model NT:
$$
\begin{align*}
y &\sim \text{Normal}(\mu, 1) \\
\mu &\sim \text{Student}(2, 10, 1) \\
\end{align*}
$$

Model TT:
$$
\begin{align*}
y &\sim \text{Student}(2, \mu, 1) \\
\mu &\sim \text{Student}(2, 10, 1) \\
\end{align*}
$$

Estimate the posterior distributions for these models and compare them. Can you explain the results using the properties of the distributions?

**Answer**


```R
data <- list(y = array(0, dim = 1))
m13m6.nn <- ulam(
    alist(
        y ~ dnorm(mu, 1),
        mu ~ dnorm(10, 1)
    ),
    data = data
    # log_lik = TRUE # this causes a failure with a warning about the dimensions - looks like it's treating the input as a real when it should be a vector
)
m13m6.tn <- ulam(
    alist(
        y ~ dstudent(2, mu, 1),
        mu ~ dnorm(10, 1)
    ),
    data = data
    # log_lik = TRUE # this causes a failure with a warning about the dimensions - looks like it's treating the input as a real when it should be a vector
)
m13m6.nt <- ulam(
    alist(
        y ~ dnorm(mu, 1),
        mu ~ dstudent(2, 10, 1)
    ),
    data = data
    # log_lik = TRUE # this causes a failure with a warning about the dimensions - looks like it's treating the input as a real when it should be a vector
)
m13m6.tt <- ulam(
    alist(
        y ~ dstudent(2, mu, 1),
        mu ~ dstudent(2, 10, 1)
    ),
    data = data
    # log_lik = TRUE # this causes a failure with a warning about the dimensions - looks like it's treating the input as a real when it should be a vector
)
```

    Running MCMC with 1 chain, with 1 thread(s) per chain...
    
    Chain 1 finished in 0.0 seconds.


```R
nn.samples.mu <- extract.samples(m13m6.nn)$mu
nt.samples.mu <- extract.samples(m13m6.nt)$mu
tn.samples.mu <- extract.samples(m13m6.tn)$mu
tt.samples.mu <- extract.samples(m13m6.tt)$mu

nn.samples.y <- rnorm(length(nn.samples.mu), nn.samples.mu, 1)
nt.samples.y <- rnorm(length(nt.samples.mu), nt.samples.mu, 1)
tn.samples.y <- rstudent(length(tn.samples.mu), 2, tn.samples.mu, 1)
tt.samples.y <- rstudent(length(tt.samples.mu), 2, tt.samples.mu, 1)

plot_df <- rbind(
    data.frame(y = nn.samples.y, model = "nn"),
    data.frame(y = nt.samples.y, model = "nt"),
    data.frame(y = tn.samples.y, model = "tn"),
    data.frame(y = tt.samples.y, model = "tt")
)
ggplot(plot_df, aes(group = model, colour = model)) +
    geom_density(aes(y))
```


    
![png](chapter_13_images/output_124_0.png)
    


1. The NN model (normal likelihood, normal prior) looks approximately normal with a mean just around the halfway point of the data and the prior, as expected. This is the kind of behaviour that we're seen often.
2. The NT model shows almost all of the weight closer to the data than to the prior. This is due to the fact that the Student-t distribution has wider tails, and so is a 'less strong' distribution; the posterior is more influenced by the data than the prior.
3. The TN model is is converse of the NT one; here, the prior is stronger than the likelihood, and so most of the mass of the posterior remains aronud the prior rather than being influenced by the data.
4. The TT model is most puzzling! I would have naively expected to see results like in the first one (NN), but instead there's a bimodal distribution with peaks around the data ($y=0$) and the prior ($y=10$). Even more interestingly, depending on the run, which peak is the largest switches. I am honestly not sure what's going on here!

**13H1** In 1980, a typical Begali woman could have 5 or more chilnder in her lifetime. By 2000, that number had reduced to only 2 or 3. We're going to look at historical data for when contraception was available but many families chose not to use it. These data are in `data(bengladesh)` and come from the 1988 Bagladesh Fertility Survey. Each row is one of the 1934 women. There are siz variables, but we are just going to focus on two of them:
1. `district`: ID of the administrative district each woman resided in
2. `use.contraception`: an indicator (0 / 1) of whether the woman was using contraception

The first thing to do is to ensure that thecluster variable, `district`, is a contiguous set of integers. Recal that these values will be index values inside the mode. If there are gaps, we'll have parameters for wheich there is no data to inform them. Worse, the model probably won't run.

Let's look at the values:


```R
data(bangladesh)
d <- bangladesh
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 6</caption>
<thead>
	<tr><th></th><th scope=col>woman</th><th scope=col>district</th><th scope=col>use.contraception</th><th scope=col>living.children</th><th scope=col>age.centered</th><th scope=col>urban</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>1</td><td>0</td><td>4</td><td> 18.4400</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>2</td><td>1</td><td>0</td><td>1</td><td> -5.5599</td><td>1</td></tr>
	<tr><th scope=row>3</th><td>3</td><td>1</td><td>0</td><td>3</td><td>  1.4400</td><td>1</td></tr>
	<tr><th scope=row>4</th><td>4</td><td>1</td><td>0</td><td>4</td><td>  8.4400</td><td>1</td></tr>
	<tr><th scope=row>5</th><td>5</td><td>1</td><td>0</td><td>1</td><td>-13.5590</td><td>1</td></tr>
	<tr><th scope=row>6</th><td>6</td><td>1</td><td>0</td><td>1</td><td>-11.5600</td><td>1</td></tr>
</tbody>
</table>




```R
sort(unique(d$district))
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li><li>6</li><li>7</li><li>8</li><li>9</li><li>10</li><li>11</li><li>12</li><li>13</li><li>14</li><li>15</li><li>16</li><li>17</li><li>18</li><li>19</li><li>20</li><li>21</li><li>22</li><li>23</li><li>24</li><li>25</li><li>26</li><li>27</li><li>28</li><li>29</li><li>30</li><li>31</li><li>32</li><li>33</li><li>34</li><li>35</li><li>36</li><li>37</li><li>38</li><li>39</li><li>40</li><li>41</li><li>42</li><li>43</li><li>44</li><li>45</li><li>46</li><li>47</li><li>48</li><li>49</li><li>50</li><li>51</li><li>52</li><li>53</li><li>55</li><li>56</li><li>57</li><li>58</li><li>59</li><li>60</li><li>61</li></ol>



District 54 is absent, which means that district is not a good index variable. Luckily we can fix that:


```R
d$district_id <- as.integer(as.factor(d$district))
sort(unique(d$district_id))
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li><li>6</li><li>7</li><li>8</li><li>9</li><li>10</li><li>11</li><li>12</li><li>13</li><li>14</li><li>15</li><li>16</li><li>17</li><li>18</li><li>19</li><li>20</li><li>21</li><li>22</li><li>23</li><li>24</li><li>25</li><li>26</li><li>27</li><li>28</li><li>29</li><li>30</li><li>31</li><li>32</li><li>33</li><li>34</li><li>35</li><li>36</li><li>37</li><li>38</li><li>39</li><li>40</li><li>41</li><li>42</li><li>43</li><li>44</li><li>45</li><li>46</li><li>47</li><li>48</li><li>49</li><li>50</li><li>51</li><li>52</li><li>53</li><li>54</li><li>55</li><li>56</li><li>57</li><li>58</li><li>59</li><li>60</li></ol>



So now we have a nice set of index variavles.

Focus on predicting `use.contraception` clustered by `district_id`. Fit both
1. A traditional fixed-effects model that uses an index variable for district, and 
2. A multilevel model with a varying intercept for district

Plot the predicted proportions of women in each district using contraception for both the fixed-effects and varying-effects models. That is, make a plot in which district ID is on the horizontal axis and expected proportion using contraception is on the vertical. Make one plot for each model, or layer them on the same plot. How do the models disagree? Can you explain the pattern of disagreement? In particular, can you explain the most extreme case of disagreement, both why they happen where they do and why the models reach different inferences?

**Answer** First, let's look at the data:


```R
plot_df <- aggregate(use.contraception ~ district_id, data = d, FUN = mean)
names(plot_df)[names(plot_df) == 'use.contraception'] <- 'prop'
# plot_df <- data.frame(
#     district_id = d$district_id,
#     prop = sapply(d$district_id, function(district_id) {
#         subset <- d[d$district_id == district_id, ]
#         sum(subset$use.contraception) / nrow(subset)
#     })
# )
head(plot_df)
```


<table class="dataframe">
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>district_id</th><th scope=col>prop</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>0.2564103</td></tr>
	<tr><th scope=row>2</th><td>2</td><td>0.3500000</td></tr>
	<tr><th scope=row>3</th><td>3</td><td>1.0000000</td></tr>
	<tr><th scope=row>4</th><td>4</td><td>0.5000000</td></tr>
	<tr><th scope=row>5</th><td>5</td><td>0.3589744</td></tr>
	<tr><th scope=row>6</th><td>6</td><td>0.2923077</td></tr>
</tbody>
</table>




```R
ggplot(plot_df, aes(district_id, prop)) +
    geom_bar(stat = 'identity')
```


    
![png](chapter_13_images/output_134_0.png)
    



```R
ggplot(plot_df, aes(prop)) +
    geom_density()
```


    
![png](chapter_13_images/output_135_0.png)
    


For the fixed-effects model, I want a mean p of about 0.5 (from the above graph). That means that I want the inverse-logit of 0.5 as my prior mean for $\alpha$.


```R
inv_logit(0.5)
```


0.622459331201855



```R
districts <- sort(unique(d$district_id))
data <- data.frame(
    district_index = districts,
    n = sapply(districts, function (district_id) {nrow(d[d$district_id == district_id & d$use.contraception == 1, ])}), # number that use contraception
    N = sapply(districts, function(district_id) {nrow(d[d$district_id == district_id, ])})# total number
)
# first the fixed-effects model
m13h1.fixed <- ulam(
    alist(
        n ~ dbinom(N, p),
        logit(p) <- alpha[district_index],
        alpha[district_index] ~ dnorm(0.622, 1)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)

# now the varying-effects model
m13h1.varying <- ulam(
    alist(
        n ~ dbinom(N, p),
        logit(p) <- alpha[district_index],
        alpha[district_index] ~ dnorm(alpha_bar, sigma),
        alpha_bar ~ dnorm(0.622, 1.5),
        sigma ~ dexp(1)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.6 seconds.
    



```R
fixed.samples.p <- extract.samples(m13h1.fixed)$p
varying.samples <- extract.samples(m13h1.varying)
varying.samples.p <- varying.samples$p

fixed.means <- apply(fixed.samples.p, 2, mean)
fixed.ci <- apply(fixed.samples.p, 2, PI)
varying.means <- apply(varying.samples.p, 2, mean)
varying.ci <- apply(varying.samples.p, 2, PI)

plot_df <- data.frame(
    district_id = districts,
    true_p = data$n / data$N,
    N = data$N,
    fixed_mean = fixed.means,
    fixed_lower = fixed.ci[1, ],
    fixed_upper = fixed.ci[2, ],
    varying_mean = varying.means,
    varying_lower = varying.ci[1, ],
    varying_upper = varying.ci[2, ]
)

ggplot(plot_df, aes(district_id, group = district_id)) +
    geom_point(aes(y = true_p, colour = "Empirical", size = N)) +
    geom_point(aes(y = fixed_mean, colour = "Fixed", size = N), position = position_jitter(width = 0.3)) +
    geom_point(aes(y = varying_mean, colour = "Varying", size = N), position = position_jitter(width = -0.3)) +
    geom_hline(aes(yintercept = mean(inv_logit(varying.samples$alpha_bar))), linetype = 'dashed') 
    
```


    
![png](chapter_13_images/output_139_0.png)
    


It's a little unclear what is going on - let's see what happens if we sort the districts by the empirical proportion using contraception.


```R
ordered_by_p <- plot_df[order(plot_df$true_p), ]
ordered_by_p$district_id <- as.character(ordered_by_p$district_id)
district_levels <- unique(ordered_by_p$district_id)
ordered_by_p$district_id <- factor(ordered_by_p$district_id, levels = district_levels)

ggplot(ordered_by_p, aes(district_id, group = district_id)) +
    geom_point(aes(y = true_p, colour = "Empirical", size = N)) +
    geom_point(aes(y = fixed_mean, colour = "Fixed", size = N), position = position_jitter(width = 0.3)) +
    geom_point(aes(y = varying_mean, colour = "Varying", size = N), position = position_jitter(width = -0.3)) +
    geom_hline(aes(yintercept = mean(inv_logit(varying.samples$alpha_bar))), linetype = 'dashed') 
```


    
![png](chapter_13_images/output_141_0.png)
    

This makes things much clearer! Unsurprisingly, the varying effects model is 'shrinking' towards the adaptive mean $\bar\alpha$, a behaviour that is not shared by the fixed effects model. Thus, for districts where the empirical proportion is below $\bar\alpha$, the varying effects model is greater than the fixed one, and for districts where the empirical proportion is above $\bar\alpha$, the varying effects model is below the fixed one. 

Of course, this effect is ameliorated by the sample size, with a larger sample reducing the amount of shrinkage (again, as expected). Thus, districts 55 and 14 have similar empirical proportions, but the varying effects model for district 14 is closer to the empirical value (less shrinkage) because the sample size is larger.

**13H2** Return to `data(Trolley)` from [[SR - Chapter 12 - Monsters and Mixtures]]. Define and fit a varying intercepts model for these data. Cluster intercepts on individual participants, as indicated by the unique values int he `id` variable. Include `action`, `intention`, and `contact` as ordinary terms. Compare the varying intercepts model and a model that ignores individuals, using both WAIC and posterior predictions. What is the impact of different stores on responses?

**Answer**


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
# This is the model from Chapter 12

data <- list(
    R = d$response,
    A = d$action,
    I = d$intention,
    C = d$contact
)

m13h2.fixed <- ulam(
    alist(
        R ~ dordlogit(phi, cutpoints),
        phi <- bA * A + bC * C + BI * I,
        BI <- bI + bIA * A + bIC * C,
        c(bA, bI, bC, bIA, bIC) ~ dnorm(0, 0.5),
        cutpoints ~ dnorm(0, 1.5)
    ),
    data = data,
    chains = 4,
    cores = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 174.8 seconds.
    Total execution time: 181.2 seconds.

```R
precis(m13h2.fixed, depth = 2)
```


<table class="dataframe">
<caption>A precis: 11 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>bIC</th><td>-1.2358126</td><td>0.09183758</td><td>-1.3869636</td><td>-1.0861686</td><td>1.001730</td><td>1094.3345</td></tr>
	<tr><th scope=row>bIA</th><td>-0.4335086</td><td>0.07666019</td><td>-0.5545407</td><td>-0.3127022</td><td>1.004693</td><td> 936.2362</td></tr>
	<tr><th scope=row>bC</th><td>-0.3437650</td><td>0.06719977</td><td>-0.4479615</td><td>-0.2310996</td><td>1.003108</td><td> 979.1471</td></tr>
	<tr><th scope=row>bI</th><td>-0.2917632</td><td>0.05528957</td><td>-0.3802125</td><td>-0.2009475</td><td>1.004477</td><td> 862.4484</td></tr>
	<tr><th scope=row>bA</th><td>-0.4724386</td><td>0.05325140</td><td>-0.5584403</td><td>-0.3876734</td><td>1.003063</td><td> 836.2431</td></tr>
	<tr><th scope=row>cutpoints[1]</th><td>-2.6342119</td><td>0.05117753</td><td>-2.7129132</td><td>-2.5496661</td><td>1.004452</td><td> 836.5497</td></tr>
	<tr><th scope=row>cutpoints[2]</th><td>-1.9377406</td><td>0.04689700</td><td>-2.0105842</td><td>-1.8613061</td><td>1.005366</td><td> 827.6363</td></tr>
	<tr><th scope=row>cutpoints[3]</th><td>-1.3431283</td><td>0.04533751</td><td>-1.4151809</td><td>-1.2699200</td><td>1.004466</td><td> 815.2432</td></tr>
	<tr><th scope=row>cutpoints[4]</th><td>-0.3082420</td><td>0.04272992</td><td>-0.3742277</td><td>-0.2394277</td><td>1.002906</td><td> 943.7041</td></tr>
	<tr><th scope=row>cutpoints[5]</th><td> 0.3622611</td><td>0.04291723</td><td> 0.2941860</td><td> 0.4329410</td><td>1.006183</td><td> 889.8002</td></tr>
	<tr><th scope=row>cutpoints[6]</th><td> 1.2684862</td><td>0.04527798</td><td> 1.1961317</td><td> 1.3441649</td><td>1.003377</td><td> 969.2096</td></tr>
</tbody>
</table>

```R
# now we add in a new varying intercept
# first we need to convert the id into an index
d$user_index <- as.integer(factor(d$id, levels = unique(d$id)))
```

```R

data <- list(
    R = d$response,
    A = d$action,
    I = d$intention,
    C = d$contact,
    user_index = d$user_index
)

m13h2.varying <- ulam(
    alist(
        R ~ dordlogit(phi, cutpoints),
        phi <- a[user_index] + bA * A + bC * C + BI * I,
        a[user_index] ~ dnorm(0, sigma_id),
        BI <- bI + bIA * A + bIC * C,
        c(bA, bI, bC, bIA, bIC) ~ dnorm(0, 0.5),
        cutpoints ~ dnorm(0, 1.5),
        sigma_id ~ dexp(1)
    ),
    data = data,
    chains = 4,
    cores = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 441.9 seconds.
    Total execution time: 475.8 seconds.

```R
pars <- c("bA", "bC", "bI")
print(precis(m13h2.fixed, depth = 2, pars = pars))
print(precis(m13h2.varying, depth = 2, pars = pars))
```

             mean         sd       5.5%      94.5%     rhat ess_bulk
    bA -0.4724386 0.05325140 -0.5584403 -0.3876734 1.003063 836.2431
    bC -0.3437650 0.06719977 -0.4479615 -0.2310996 1.003108 979.1471
    bI -0.2917632 0.05528957 -0.3802125 -0.2009475 1.004477 862.4484
             mean         sd       5.5%      94.5%     rhat ess_bulk
    bA -0.6488900 0.05658312 -0.7402288 -0.5631944 1.002245 1329.534
    bC -0.4537379 0.06910934 -0.5666902 -0.3429041 1.001486 1506.711
    bI -0.3842588 0.05897703 -0.4799357 -0.2917684 1.000902 1258.039


Even just looking at these parameters, we can see that there's a big difference! Adding in the user-specific parameter had a big effect.


```R
compare(m13h2.fixed, m13h2.varying)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m13h2.varying</th><td>31058.68</td><td>179.36508</td><td>   0.000</td><td>      NA</td><td>356.63988</td><td>1</td></tr>
	<tr><th scope=row>m13h2.fixed</th><td>36929.22</td><td> 80.72734</td><td>5870.542</td><td>173.5561</td><td> 10.95863</td><td>0</td></tr>
</tbody>
</table>



So the varying effects model is quite a bit better; it turns out that the individuals taking part in the study have an important effect on the results!

**M13H3** The `Trolley` data are also clustered by `story`, which indicates a unique narrative for each vignette. Define and fit a cross-classifyied varying intercepts model with both `id` and `story`. Use the same ordinary terms as in the previous problem. Compare this model to the previous models. What do you inder about the impact of different stories on responses?

**Answer**


```R
data <- list(
    R = d$response,
    A = d$action,
    I = d$intention,
    C = d$contact,
    user_index = d$user_index,
    story_index = as.integer(factor(d$story, levels = unique(d$story)))
)
```


```R
# cross-classified model including the story
m13h3.varying <- ulam(
    alist(
        R ~ dordlogit(phi, cutpoints),
        phi <- a[user_index] + s[story_index] + bA * A + bC * C + BI * I,
        a[user_index] ~ dnorm(0, sigma_id),
        s[story_index] ~ dnorm(0, sigma_story),
        BI <- bI + bIA * A + bIC * C,
        c(bA, bI, bC, bIA, bIC) ~ dnorm(0, 0.5),
        cutpoints ~ dnorm(0, 1.5),
        sigma_id ~ dexp(1),
        sigma_story ~ dexp(1)
    ),
    data = data,
    chains = 4,
    cores = 4,
    log_lik = TRUE
)

```

    Running MCMC with 4 parallel chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 619.7 seconds.
    Total execution time: 668.8 seconds.

```R
params <- c("bC", "bI", "bA", "sigma_id", "sigma_story")
plot(coeftab(m13h2.fixed, m13h2.varying, m13h3.varying), pars = params)
```

    Error in h(simpleError(msg, call)): error in evaluating the argument 'x' in selecting a method for function 'plot': object 'm13h2.fixed' not found
    Traceback:


    1. plot(coeftab(m13h2.fixed, m13h2.varying, m13h3.varying), pars = params)

    2. coeftab(m13h2.fixed, m13h2.varying, m13h3.varying)

    3. .handleSimpleError(function (cond) 
     . .Internal(C_tryCatchHelper(addr, 1L, cond)), "object 'm13h2.fixed' not found", 
     .     base::quote(eval(expr, envir, enclos)))

    4. h(simpleError(msg, call))



```R
compare(m13h2.fixed, m13h2.varying, m13h3.varying)
```


<table class="dataframe">
<caption>A compareIC: 3 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m13h3.varying</th><td>30569.40</td><td>180.32869</td><td>   0.0000</td><td>       NA</td><td>367.60062</td><td> 1.000000e+00</td></tr>
	<tr><th scope=row>m13h2.varying</th><td>31058.68</td><td>179.36508</td><td> 489.2737</td><td> 42.57966</td><td>356.63988</td><td>5.695835e-107</td></tr>
	<tr><th scope=row>m13h2.fixed</th><td>36929.22</td><td> 80.72734</td><td>6359.8160</td><td>175.28190</td><td> 10.95863</td><td> 0.000000e+00</td></tr>
</tbody>
</table>



So the one including the effect of the story is definitely doing better than the others.


```R
# let's take a look at the different sigmas
samples <- extract.samples(m13h3.varying)
```


```R
plot_df <- rbind(
    data.frame(x = samples$sigma_id, type = "Person"),
    data.frame(x = samples$sigma_story, type = "Story")
)

ggplot(plot_df, aes(x, colour = type)) +
    geom_density()
```


    
![png](chapter_13_images/output_160_0.png)
    


Interesting! So it looks like there is much more variability in the people than in the stories! That is, while including the story effect is important, different people tend to react relatively the same to each different story. This is especially interesting because we separately pull out the I, A, and C components, so there's something about the different stories that is having an effect (albeit not a huge one) *over and above* the components that we're interested in.

**13H4** Revisit the Reed frog survival data, `data(reedFrogs)`, and add the `predation` and `size` treatment variables to the varying intercepts model. Consider models with either predictor alone, both predictors, as well as a model including their interaction. What do you infer about the causal influense of these predictor variables? Also focus on the inferred variation across tanks (the $\sigma$ across tanks). Explain why it changes as it does across models with different predictors included. 

**Answer** This is largely the same as 13M1, but now we're focusing on the causal influence rather than just the variation. We'll run the models again and take a look at some data.


```R
data(reedfrogs)
d <- reedfrogs
d$tank <- 1:nrow(d)
d$pred_index <- ifelse(d$pred == 'no', 1L, 2L)
d$size_index <- ifelse(d$size == 'small', 1L, 2L)
d$pred_size_index <- 2 * (d$pred_index - 1) + (d$size_index - 1) + 1

data <- list(
    S = d$surv,
    N = d$density,
    tank = d$tank,
    pred_index = d$pred_index,
    size_index = d$size_index,
    pred_size_index = d$pred_size_index
)
```


```R
m13h4.base <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)

# predation
m13h4.pred <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + b[pred_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        b[pred_index] ~ dnorm(0, 1.5)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)

# size
m13h4.size <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + g[size_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        g[size_index] ~ dnorm(0, 1.5)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)

# predation and size (no interaction)
m13h4.pred_size_no_interaction <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + b[pred_index] + g[size_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        b[pred_index] ~ dnorm(0, 1.5),
        g[size_index] ~ dnorm(0, 1.5)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)

# predation and size (interaction)
m13h4.pred_size_interaction <- ulam(
    alist(
        S ~ dbinom(N, p),
        logit(p) <- a[tank] + b[pred_index] + g[size_index] + e[pred_size_index],
        a[tank] ~ dnorm(a_bar, sigma),
        a_bar ~ dnorm(0, 1.5),
        sigma ~ dexp(1),
        b[pred_index] ~ dnorm(0, 1.5),
        g[size_index] ~ dnorm(0, 1.5),
        e[pred_size_index] ~ dnorm(0, 1.5)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)

```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.5 seconds.
    Total execution time: 2.2 seconds.

```R
par(bg = 'white')
params <- c('sigma')
plot(coeftab(m13h4.base, m13h4.pred, m13h4.size, m13h4.pred_size_no_interaction, m13h4.pred_size_interaction), pars = params)
```


    
![png](chapter_13_images/output_165_0.png)
    



```R
compare(m13h4.base, m13h4.pred, m13h4.size, m13h4.pred_size_no_interaction, m13h4.pred_size_interaction)
```


<table class="dataframe">
<caption>A compareIC: 5 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m13h4.base</th><td>199.0502</td><td>7.350988</td><td>0.0000000</td><td>      NA</td><td>20.42246</td><td>0.30098895</td></tr>
	<tr><th scope=row>m13h4.pred</th><td>199.4221</td><td>9.179128</td><td>0.3719159</td><td>5.759817</td><td>19.31676</td><td>0.24991369</td></tr>
	<tr><th scope=row>m13h4.pred_size_no_interaction</th><td>200.0441</td><td>8.973249</td><td>0.9938571</td><td>6.157536</td><td>19.16602</td><td>0.18312061</td></tr>
	<tr><th scope=row>m13h4.pred_size_interaction</th><td>200.1870</td><td>9.670286</td><td>1.1368335</td><td>6.891638</td><td>19.27705</td><td>0.17048662</td></tr>
	<tr><th scope=row>m13h4.size</th><td>201.3463</td><td>7.253817</td><td>2.2961013</td><td>1.307723</td><td>21.37179</td><td>0.09549013</td></tr>
</tbody>
</table>




```R
par(bg = 'white')
params <- c('b[1]', 'b[2]', 'g[1]', 'g[2]')
plot(coeftab(m13h4.base, m13h4.pred, m13h4.size, m13h4.pred_size_no_interaction, m13h4.pred_size_interaction), pars = params)
```


    
![png](chapter_13_images/output_167_0.png)
    


All of these models do roughly as good a job, regardless of the variables that we include. From this, is looks like the individual variation in the tanks is the most important causal variable, followed by the inclusion of predation (although with only a small effect). The size of the tank seems to have very little effect on the outcome.
