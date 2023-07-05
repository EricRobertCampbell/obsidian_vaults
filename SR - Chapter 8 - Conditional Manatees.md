# Chapter 8 - Conditional Manatees

Lots of manatees have scars from boat propeller. Similarly, lots of WWII bombers that came back had holes from artillery or whatnot. In both cases, we'd like to reduce them damage. Naively, we should put propeller guards on boats and improve the armour where we find lots of holes.

In both cases, this is misleading.

Autopsies of manatees show that it is collisions with the blunt parts of boats (e.g. keel) that cause the most deaths. Similarly, you should increase the armour on the sections of the plane that *don't* show damage, because any of the hits there result in the plane not returning.

In both cases the evidence is misleading because it is conditional on *survival*.

**Conditioning** is one of te most important principles of statistical inference. Data are conditional on how they got into our sample! All model-based inference is conditional on the model!

Our models so far assume a degree of independence - we don't have a way to ask about, say, the effect of milk energy on brain size conditional on taxonomic group.

To model deeper conditionality - where one *predictor* depends on another *predictor* - we need **interaction** (aka **moderation**). This is a way of alloweing parameters (or more specifically, their posterior distributions) to be conditional on other aspects of the data.

More complex models (Generalized Linear Models, Multilevel models) all depend on interactions. In this chapter, we'll look at a simple example to get a feel for the idea.

## Building an Interaction

For most countries, the ruggedness of the terrain is negatively associated with log GDP. However, in Africa it is the opposite. There may be some causal reason for this - for instance, more rugged countries were more protected from the slave trade.

The DAG for this could look as follows:


```R
library('ggplot2')
library('rethinking')
library('dagitty')
```


```R
dag <- dagitty('dag{
    U [latent, pos="0,0"]
    R [pos="-1,-1"]
    G [pos="0,-1"]
    C [pos="1,-1"]
    U -> R -> G <- C
    U -> G
}')
drawdag(dag)
```


    
![png](sr-ch08-output_2_0.png)
    


where
- $R$ is the ruggedness
- $G$ is GPD
- $C$ is the continent
- $U$ is some unobserved confounds like distance to the coast

We'll ignore $U$ for now.

The implication of this DAG is that $R$ and $C$ both affect $G$. They could do so independently or they could interact (one moderates the influence of the other). The DAG doesn't actually specify - it just says that $C = f(R, C)$; it doesn't specify the shape of the function.

So how can we make a model that produces the conditionality (ruggedness is negatively associated with GDP outside of Africa but is positively associated within it)?

We could split the dataframe into two: one for Africa and one for the rest of the world. That's a bad idea for a few reasons.
1. Some parameters, such as $\sigma$, don't (or shouldn't) depend on the continued. By splitting, you hurt the accuracy of the prediction of these parameters by essentially creating two less-ccurate estimates instead of one more-accurate one.
1. In rder to acquire probability statements about the variable we used to split (`cont_africa`), we need to include it in the model. But now, since we split on it, we can't assess any uncertainy about the predictive value of distinguishing between African and non-African countries.
1. We may want to use information criteria to assess the different models. We can only do that if the different models were trained on the same data, which would not be the case if we split this into two data frames
1. Once we start to use multilevel models, it turns out that there are advantages to 'borrowing' information across categories (like 'Africa' and 'not Africa'). This is especially true when sample sizes vary across categories, such that overfitting risk is higher within some categories.

### Making a rugged model

We'll start with a single model, ignoring continent.


```R
data(rugged)
d <- rugged

# log the gdp
d$log_gdp <- log(d$rgdppc_2000)

# only get the countries with GDP data
dd <- d[complete.cases(d$rgdppc_2000),]

# rescale the variables
mean_gdp <- mean(dd$log_gdp)
dd$log_gdp_std <- dd$log_gdp / mean_gdp
dd$rugged_std <- dd$rugged / max(dd$rugged)

head(dd)
```


<table class="dataframe">
<caption>A data.frame: 6 × 54</caption>
<thead>
	<tr><th></th><th scope=col>isocode</th><th scope=col>isonum</th><th scope=col>country</th><th scope=col>rugged</th><th scope=col>rugged_popw</th><th scope=col>rugged_slope</th><th scope=col>rugged_lsd</th><th scope=col>rugged_pc</th><th scope=col>land_area</th><th scope=col>lat</th><th scope=col>⋯</th><th scope=col>slave_exports</th><th scope=col>dist_slavemkt_atlantic</th><th scope=col>dist_slavemkt_indian</th><th scope=col>dist_slavemkt_saharan</th><th scope=col>dist_slavemkt_redsea</th><th scope=col>pop_1400</th><th scope=col>european_descent</th><th scope=col>log_gdp</th><th scope=col>log_gdp_std</th><th scope=col>rugged_std</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>⋯</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>3</th><td>AGO</td><td> 24</td><td>Angola              </td><td>0.858</td><td>0.714</td><td> 2.274</td><td>0.228</td><td> 4.906</td><td>124670</td><td>-12.299</td><td>⋯</td><td>3610000</td><td>5.669</td><td>6.981</td><td>4.926</td><td>3.872</td><td>1223208</td><td>  2.000</td><td>7.492609</td><td>0.8797119</td><td>0.1383424702</td></tr>
	<tr><th scope=row>5</th><td>ALB</td><td>  8</td><td>Albania             </td><td>3.427</td><td>1.597</td><td>10.451</td><td>1.006</td><td>62.133</td><td>  2740</td><td> 41.143</td><td>⋯</td><td>      0</td><td>   NA</td><td>   NA</td><td>   NA</td><td>   NA</td><td> 200000</td><td>100.000</td><td>8.216929</td><td>0.9647547</td><td>0.5525636891</td></tr>
	<tr><th scope=row>8</th><td>ARE</td><td>784</td><td>United Arab Emirates</td><td>0.769</td><td>0.316</td><td> 2.112</td><td>0.191</td><td> 6.142</td><td>  8360</td><td> 23.913</td><td>⋯</td><td>      0</td><td>   NA</td><td>   NA</td><td>   NA</td><td>   NA</td><td>  19200</td><td>  0.000</td><td>9.933263</td><td>1.1662705</td><td>0.1239922606</td></tr>
	<tr><th scope=row>9</th><td>ARG</td><td> 32</td><td>Argentina           </td><td>0.775</td><td>0.220</td><td> 2.268</td><td>0.226</td><td> 9.407</td><td>273669</td><td>-35.396</td><td>⋯</td><td>      0</td><td>   NA</td><td>   NA</td><td>   NA</td><td>   NA</td><td> 276632</td><td> 89.889</td><td>9.407032</td><td>1.1044854</td><td>0.1249596904</td></tr>
	<tr><th scope=row>10</th><td>ARM</td><td> 51</td><td>Armenia             </td><td>2.688</td><td>0.934</td><td> 8.178</td><td>0.799</td><td>50.556</td><td>  2820</td><td> 40.294</td><td>⋯</td><td>      0</td><td>   NA</td><td>   NA</td><td>   NA</td><td>   NA</td><td> 105743</td><td>  0.500</td><td>7.792343</td><td>0.9149038</td><td>0.4334085779</td></tr>
	<tr><th scope=row>12</th><td>ATG</td><td> 28</td><td>Antigua and Barbuda </td><td>0.006</td><td>0.003</td><td> 0.012</td><td>0.003</td><td> 0.000</td><td>    44</td><td> 17.271</td><td>⋯</td><td>      0</td><td>   NA</td><td>   NA</td><td>   NA</td><td>   NA</td><td>    747</td><td>     NA</td><td>9.212541</td><td>1.0816501</td><td>0.0009674299</td></tr>
</tbody>
</table>



Each row is a country.

Normally we would rescale by making each into a z-score. However, we want these values (ruggedness and gpd) to be relevant to humans. That means that we score the ruggedness from 0 (totally flat) to 1 (the most rugged country, which happens to be Losotho). The GDP is similarly scale to be a proportion of the average.

Model:

$$
\begin{align*}
\log (y_i) &\sim \text{Normal}(\mu_i, sigma) \\
\mu_i &= \alpha + \beta(r_i - \bar{r})
\end{align*}
$$

Now we need the priors. I don't know much a priori about the relationship betwen ruggedness and GPD. Luckily, we can lean on the data a bit.

Let's look at the intercept $\alpha$. This is the log GDP when the ruggedness is at the mean. This should be close to 1, since we scaled it that way. Let's guess at
$$
\alpha \sim \text{Normal}(1, 1)
$$

For $\beta$, we can start with a mean of 0 (no positive or negative bias). What about the standard deviation? Again, we'll guess at 1. Then
$$
\beta \sim \text{Normal}(0, 1)
$$

We'll also take a broad guess at $\sigma$:

$$
\sigma \sim \text{Exponential}(1)
$$


```R
m8.1 <- quap(
    alist(
        log_gdp_std ~ dnorm(mu, sigma),
        mu <- a + b * (rugged_std - 0.215),
        a ~ dnorm(1, 1),
        b ~ dnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data=dd
)
```


```R
# prior predictive check
set.seed(7)
prior <- extract.prior(m8.1)

# grab 50 lines from the prior
rugged_seq <- seq(from=-0.1, to=1.1, length.out=30)
mu <- link(m8.1, post=prior, data=data.frame(rugged_std=rugged_seq))
```


```R
lines_df <- data.frame(rugged_std=double(), log_gdp_std=double(), group=integer())

for ( i in 1:50 ) {
    new_lines_df = data.frame(rugged_std=rugged_seq, log_gdp_std=mu[i,], group=i)
    lines_df <- rbind(lines_df, new_lines_df)
}
lines_df$group <- as.factor(lines_df$group)
ggplot() +
    geom_point(data=dd, aes(rugged_std, log_gdp_std)) + # data
    geom_line(data=lines_df, aes(rugged_std, log_gdp_std, group=group), alpha=0.5) + # regression lines
    geom_hline(yintercept = min(dd$log_gdp_std), linetype='dotted', colour='red') +
    geom_hline(yintercept = max(dd$log_gdp_std), linetype='dotted', colour='red') +
    xlab("Standardized Ruggedness") +
    ylab("Standardized Log GDP")

```


    
![png](sr-ch08-output_8_0.png)
    


As you can see, this is some nonsense. We need a tighter standard deviation on the $\alpha$ prior. Maybe something like $\sigma \sim \text{Normal}(0, 0.1)$? This would put most of the weight between the observed values (0.8 - 1.2).

Also, the slopes are too variable. It is just not plausible that *most* of the variation that we see is due to ruggedness. We need to tighten up the $\beta$ prior as well. Maybe $\beta \sim \text{Normal}(0, 0.3)$?


```R
m8.1 <- quap(
    alist(
        log_gdp_std ~ dnorm(mu, sigma),
        mu <- a + b * (rugged_std - 0.215),
        a ~ dnorm(1, 0.1),
        b ~ dnorm(0, 0.3),
        sigma ~ dexp(1)
    ),
    data=dd
)

set.seed(7)
prior <- extract.prior(m8.1)

# grab 50 lines from the prior
rugged_seq <- seq(from=-0.1, to=1.1, length.out=30)
mu <- link(m8.1, post=prior, data=data.frame(rugged_std=rugged_seq))

lines_df <- data.frame(rugged_std=double(), log_gdp_std=double(), group=integer())

for ( i in 1:50 ) {
    new_lines_df = data.frame(rugged_std=rugged_seq, log_gdp_std=mu[i,], group=i)
    lines_df <- rbind(lines_df, new_lines_df)
}
lines_df$group <- as.factor(lines_df$group)
ggplot() +
    geom_point(data=dd, aes(rugged_std, log_gdp_std)) + # data
    geom_line(data=lines_df, aes(rugged_std, log_gdp_std, group=group), alpha=0.5) + # regression lines
    geom_hline(yintercept = min(dd$log_gdp_std), linetype='dotted', colour='red') +
    geom_hline(yintercept = max(dd$log_gdp_std), linetype='dotted', colour='red') +
    xlab("Standardized Ruggedness") +
    ylab("Standardized Log GDP")
```


    
![png](sr-ch08-output_10_0.png)
    


This seems much more plausible.


```R
precis(m8.1)
```


<table class="dataframe">
<caption>A precis: 3 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>0.999999515</td><td>0.010411972</td><td> 0.9833592</td><td>1.01663986</td></tr>
	<tr><th scope=row>b</th><td>0.001990935</td><td>0.054793464</td><td>-0.0855796</td><td>0.08956147</td></tr>
	<tr><th scope=row>sigma</th><td>0.136497402</td><td>0.007396152</td><td> 0.1246769</td><td>0.14831788</td></tr>
</tbody>
</table>



From this, we see no association between terrain ruggedness and log GPD.

### Adding an indicator variable isn't enough

Just adding an indicator variable, `cont_africa`, isn't enough. Let's see that not working!

We need to make it so that the model has two intercepts: one for in Africa and one for out.

$$
\mu_i = \alpha_{CID[i]} + \beta(r_i - \bar{r})
$$

where $CID[i]$ is the continent indicator ($i$ is 1 or 2) for the $i$th row.


```R
# variable to index Africa (1) or not (2)
dd$cid <- ifelse(dd$cont_africa==1, 1, 2)
```


```R
m8.2 <- quap(
    alist(
        log_gdp_std ~ dnorm(mu, sigma),
        mu <- a[cid] + b * (rugged_std - 0.215),
        a[cid] ~ dnorm(1, 0.1),
        b ~ dnorm(0, 0.3),
        sigma ~ dexp(1)
    ),
    data=dd
)
```


```R
# compare using WAIC
compare(m8.1, m8.2)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m8.2</th><td>-252.2687</td><td>15.30518</td><td> 0.00000</td><td>      NA</td><td>4.258517</td><td>1.000000e+00</td></tr>
	<tr><th scope=row>m8.1</th><td>-188.7542</td><td>13.29295</td><td>63.51448</td><td>15.14678</td><td>2.690401</td><td>1.614382e-14</td></tr>
</tbody>
</table>



The standard error of the weight is 15, and the actual difference is 63.5 - this is pretty important! It looks like the continent indicator is picking up some important information.


```R
precis(m8.2, depth=2)
```


<table class="dataframe">
<caption>A precis: 4 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td> 0.88041284</td><td>0.015937003</td><td> 0.8549424</td><td>0.90588325</td></tr>
	<tr><th scope=row>a[2]</th><td> 1.04916425</td><td>0.010185554</td><td> 1.0328858</td><td>1.06544274</td></tr>
	<tr><th scope=row>b</th><td>-0.04651347</td><td>0.045686725</td><td>-0.1195297</td><td>0.02650274</td></tr>
	<tr><th scope=row>sigma</th><td> 0.11238738</td><td>0.006091077</td><td> 0.1026527</td><td>0.12212209</td></tr>
</tbody>
</table>




```R
plot(precis(m8.2, depth=2))
```


    
![png](sr-ch08-output_19_0.png)
    


The parameter $a[1]$ is for African nations and seems reliably lower than the one for non-African nations. The posterior contrast between the two is


```R
post <- extract.samples(m8.2)
diff_a1_a2 <- post$a[,1] - post$a[,2]
PI(diff_a1_a2)
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>5%</dt><dd>-0.199011817683046</dd><dt>94%</dt><dd>-0.137848971367759</dd></dl>



So the difference seems reliably negative. Now let's graph this to see what the effect looks like.


```R
rugged_seq <- seq(from=-0.1, to=1.1, length.out=30)
# compute the means over the samples, fixing CID=1 and then 2
mu.Africa <- link(m8.2, data=data.frame(cid=1, rugged_std=rugged_seq))
mu.NotAfrica <- link(m8.2, data=data.frame(cid=2, rugged_std=rugged_seq))

# summarize the means and intervals
mu.NotAfrica_mu <- apply(mu.NotAfrica, 2, mean)
mu.NotAfrica_ci <- apply(mu.NotAfrica, 2, PI, prob=0.97)
mu.Africa_mu <- apply(mu.Africa, 2, mean)
mu.Africa_ci <- apply(mu.Africa, 2, PI, prob=0.97)
```


```R
dd$cont_africa <- as.factor(dd$cont_africa)
ggplot() +
    geom_point(data=dd, aes(rugged_std, log_gdp_std, colour=cont_africa)) +
    geom_line(data=data.frame(rugged_std=rugged_seq, log_gpd_std=mu.NotAfrica_mu), aes(rugged_std, log_gpd_std)) +
    geom_ribbon(data=data.frame(rugged_std=rugged_seq, lower=mu.NotAfrica_ci[1,], upper=mu.NotAfrica_ci[2,]), aes(
        rugged_std, ymin=lower, ymax=upper, alpha=0.2
    )) +
    geom_line(data=data.frame(rugged_std=rugged_seq, log_gpd_std=mu.Africa_mu), aes(rugged_std, log_gpd_std)) +
    geom_ribbon(data=data.frame(rugged_std=rugged_seq, lower=mu.Africa_ci[1,], upper=mu.Africa_ci[2,]), aes(
        rugged_std, ymin=lower, ymax=upper, alpha=0.2
    )) +
    xlab('Ruggedness') +
    ylab("Log GDP")
```


    
![png](sr-ch08-output_24_0.png)
    


Because we only allowed the intercept to change, the slopes are identical. However, we do see a lower GPD level in the African countries. So how can we add the interaction to recover the change in slope that we saw earlier? We'll just double down and allow the slope to also be indexed to the continent indicator!

$$
\mu_u = \alpha_{CID[i]} + \beta_{CID[i]} * (r_i - \bar{r})
$$


```R
m8.3 <- quap(
    alist(
        log_gdp_std ~ dnorm(mu, sigma),
        mu <- a[cid] + b[cid] * (rugged_std - 0.215),
        a[cid] ~ dnorm(1, 0.1),
        b[cid] ~ dnorm(0, 0.3),
        sigma ~ dexp(1)
    ),
    data=dd
)
```


```R
precis(m8.3, depth=2)
```


<table class="dataframe">
<caption>A precis: 5 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td> 0.8865442</td><td>0.015676378</td><td> 0.86149028</td><td> 0.91159804</td></tr>
	<tr><th scope=row>a[2]</th><td> 1.0505689</td><td>0.009937071</td><td> 1.03468758</td><td> 1.06645030</td></tr>
	<tr><th scope=row>b[1]</th><td> 0.1326132</td><td>0.074207629</td><td> 0.01401504</td><td> 0.25121129</td></tr>
	<tr><th scope=row>b[2]</th><td>-0.1427253</td><td>0.054751746</td><td>-0.23022921</td><td>-0.05522148</td></tr>
	<tr><th scope=row>sigma</th><td> 0.1094993</td><td>0.005935989</td><td> 0.10001242</td><td> 0.11898613</td></tr>
</tbody>
</table>



So here we see that the slopes are nerly reversed: 0.13 vs. -0.14. Let's use PSIS to compare the models. We use PSIS because of the Pareto-$k$ warnings.


```R
compare(m8.1, m8.2, m8.3, func=PSIS)
```

    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    
    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    



<table class="dataframe">
<caption>A compareIC: 3 × 6</caption>
<thead>
	<tr><th></th><th scope=col>PSIS</th><th scope=col>SE</th><th scope=col>dPSIS</th><th scope=col>dSE</th><th scope=col>pPSIS</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m8.3</th><td>-259.1326</td><td>15.21911</td><td> 0.00000</td><td>       NA</td><td>5.166182</td><td>9.716174e-01</td></tr>
	<tr><th scope=row>m8.2</th><td>-252.0662</td><td>15.40450</td><td> 7.06637</td><td> 6.670773</td><td>4.340923</td><td>2.838263e-02</td></tr>
	<tr><th scope=row>m8.1</th><td>-188.5944</td><td>13.37393</td><td>70.53821</td><td>15.453690</td><td>2.750823</td><td>4.680775e-16</td></tr>
</tbody>
</table>



The model m8.3 has almost all of the weight (97%), which is very strong support for including the interaction (as long as our goal is prediction). The slight weight given to m8.2 indicates that the slopes in m8.3 are slightly overfit. And notice that the difference in the top two models (6.7) is basically the same as the standard error (7.5). Let's look at the pareto $k$ values.


```R
plot(PSIS(m8.3, pointwise=T)$k)
```


    
![png](sr-ch08-output_31_0.png)
    


This might be a place to use robust regression (like the Student-t one that we used in the previous chapter).


```R
rugged_seq <- seq(from=-0.1, to=1.1, length.out=30)
# compute the means over the samples, fixing CID=1 and then 2
mu.Africa <- link(m8.3, data=data.frame(cid=1, rugged_std=rugged_seq))
mu.NotAfrica <- link(m8.3, data=data.frame(cid=2, rugged_std=rugged_seq))

# summarize the means and intervals
mu.NotAfrica_mu <- apply(mu.NotAfrica, 2, mean)
mu.NotAfrica_ci <- apply(mu.NotAfrica, 2, PI, prob=0.97)
mu.Africa_mu <- apply(mu.Africa, 2, mean)
mu.Africa_ci <- apply(mu.Africa, 2, PI, prob=0.97)

ggplot() +
    geom_point(data=dd, aes(rugged_std, log_gdp_std, colour=cont_africa)) +
    geom_line(data=data.frame(rugged_std=rugged_seq, log_gpd_std=mu.NotAfrica_mu), aes(rugged_std, log_gpd_std)) +
    geom_ribbon(data=data.frame(rugged_std=rugged_seq, lower=mu.NotAfrica_ci[1,], upper=mu.NotAfrica_ci[2,]), aes(
        rugged_std, ymin=lower, ymax=upper, alpha=0.2
    )) +
    geom_line(data=data.frame(rugged_std=rugged_seq, log_gpd_std=mu.Africa_mu), aes(rugged_std, log_gpd_std)) +
    geom_ribbon(data=data.frame(rugged_std=rugged_seq, lower=mu.Africa_ci[1,], upper=mu.Africa_ci[2,]), aes(
        rugged_std, ymin=lower, ymax=upper, alpha=0.2
    )) +
    xlab('Ruggedness') +
    ylab("Log GDP")
```


    
![png](sr-ch08-output_33_0.png)
    


So here we can see the recovered slopes are different - negative outside Africa but positive within it!

## Symmetry of Interactions

There are two symmetrical ways of interpreting what we just did:
1. How much does the association between ruggedness and log GDP depend on whether the nation is in Africa?
1. How much does the association of Africa with log GDP depend on ruggedness?

Let's look at the definition of $\mu_i$ again:

$$
\mu_i = \alpha_{CID[i]} + \beta_{CID[i]}(r_i - \bar{r})
$$

The way we've interpreted this is as the slope being conditional on the continent. To see the other way, let's re-write this slightly:

$$
\mu_i = \underbrace{(2 - \text{CID})(\alpha_1 + \beta_1 (r_i - \bar{r}))}_{\text{$CID[i]$ = 1}} + \underbrace{(1 - \text{CID})(\alpha_2 + \beta_2 (r_i - \bar{r}))}_{\text{$CID[i]$ = 2}}
$$

This is actually the same model, but now it's more aparent that this depends on the ruggedness. Now if we imagine switching a nation to Africa, in order to know what this does for the prediction, we have to know the ruggedness.

Let's plot the reverse interpretation: *the association of being in Africa with log GDP depends upon terrain ruggedness*. We'll computer the difference between a nation in Africa and one outside it and display the results.


```R
rugged_seq <- seq(from=-0.2, to=1.2, length.out=30)
muA <- link(m8.3, data=data.frame(cid=1, rugged_std=rugged_seq))
muN <- link(m8.3, data=data.frame(cid=2, rugged_std=rugged_seq))
delta <- muA - muN
head(delta)
delta_mu <- apply(delta, 2, mean)
delta_ci <- apply(delta, 2, PI, prob=0.97)
```


<table class="dataframe">
<caption>A matrix: 6 × 30 of type dbl</caption>
<tbody>
	<tr><td>-0.2718117</td><td>-0.2589063</td><td>-0.2460009</td><td>-0.2330955</td><td>-0.2201901</td><td>-0.2072847</td><td>-0.1943793</td><td>-0.1814739</td><td>-0.1685685</td><td>-0.1556631</td><td>⋯</td><td>-0.013703740</td><td>-0.0007983435</td><td> 0.01210705</td><td> 0.02501245</td><td> 0.03791785</td><td> 0.05082324</td><td> 0.063728639</td><td>0.076634036</td><td>0.08953943</td><td>0.10244483</td></tr>
	<tr><td>-0.3282910</td><td>-0.3098404</td><td>-0.2913898</td><td>-0.2729392</td><td>-0.2544886</td><td>-0.2360380</td><td>-0.2175874</td><td>-0.1991368</td><td>-0.1806863</td><td>-0.1622357</td><td>⋯</td><td> 0.040720823</td><td> 0.0591714132</td><td> 0.07762200</td><td> 0.09607259</td><td> 0.11452318</td><td> 0.13297377</td><td> 0.151424362</td><td>0.169874952</td><td>0.18832554</td><td>0.20677613</td></tr>
	<tr><td>-0.3597178</td><td>-0.3420106</td><td>-0.3243034</td><td>-0.3065962</td><td>-0.2888891</td><td>-0.2711819</td><td>-0.2534747</td><td>-0.2357675</td><td>-0.2180603</td><td>-0.2003531</td><td>⋯</td><td>-0.005574033</td><td> 0.0121331551</td><td> 0.02984034</td><td> 0.04754753</td><td> 0.06525472</td><td> 0.08296191</td><td> 0.100669098</td><td>0.118376287</td><td>0.13608348</td><td>0.15379066</td></tr>
	<tr><td>-0.2162713</td><td>-0.2080968</td><td>-0.1999224</td><td>-0.1917479</td><td>-0.1835735</td><td>-0.1753990</td><td>-0.1672245</td><td>-0.1590501</td><td>-0.1508756</td><td>-0.1427011</td><td>⋯</td><td>-0.052782044</td><td>-0.0446075813</td><td>-0.03643312</td><td>-0.02825866</td><td>-0.02008419</td><td>-0.01190973</td><td>-0.003735267</td><td>0.004439196</td><td>0.01261366</td><td>0.02078812</td></tr>
	<tr><td>-0.3223655</td><td>-0.3041704</td><td>-0.2859752</td><td>-0.2677801</td><td>-0.2495850</td><td>-0.2313899</td><td>-0.2131947</td><td>-0.1949996</td><td>-0.1768045</td><td>-0.1586094</td><td>⋯</td><td> 0.041537000</td><td> 0.0597321247</td><td> 0.07792725</td><td> 0.09612237</td><td> 0.11431750</td><td> 0.13251262</td><td> 0.150707747</td><td>0.168902871</td><td>0.18709800</td><td>0.20529312</td></tr>
	<tr><td>-0.2977979</td><td>-0.2800330</td><td>-0.2622681</td><td>-0.2445032</td><td>-0.2267383</td><td>-0.2089734</td><td>-0.1912085</td><td>-0.1734436</td><td>-0.1556786</td><td>-0.1379137</td><td>⋯</td><td> 0.057500325</td><td> 0.0752652390</td><td> 0.09303015</td><td> 0.11079507</td><td> 0.12855998</td><td> 0.14632489</td><td> 0.164089807</td><td>0.181854721</td><td>0.19961963</td><td>0.21738455</td></tr>
</tbody>
</table>




```R
plot_df <- data.frame(
    rugged=rugged_seq,
    log_gdp_delta=delta_mu,
    lower=delta_ci[1,],
    upper=delta_ci[2,]
)
ggplot(plot_df, aes(rugged)) +
    geom_line(aes(y=log_gdp_delta)) +
    geom_ribbon(aes(ymin=lower, ymax=upper, alpha=0.2)) +
    geom_abline(aes(slope=0, intercept=0), linetype='dotted') +
    annotate('text', x=0, y=0.05, label="Africa higher GPD") +
    annotate('text', x=0, y=-0.05, label="Africa lower GPD") +
    xlab("Ruggedness") +
    ylab("log GDP delta")
```


    
![png](sr-ch08-output_36_0.png)
    


From this, we can see that it's really only at very high levels of ruggedness thata  country would be better to be in Africa.

So really, there's no difference between these interpretations - they are the same. Use the one that feels more natural.

## Continuous Interaction

So far we've been dealing with a discrete interaction: the continent. However, interactions in general are very difficult to interpret. We'll take a look at constructing and interpreting a continuous interaction.

### A Winter Flower

The data in this example are te sizes of blooms from begs of tulips grown in a greenhouse under different soil and light conditions.


```R
data(tulips)
d <- tulips
str(d)
```

    'data.frame':	27 obs. of  4 variables:
     $ bed   : Factor w/ 3 levels "a","b","c": 1 1 1 1 1 1 1 1 1 2 ...
     $ water : int  1 1 1 2 2 2 3 3 3 1 ...
     $ shade : int  1 2 3 1 2 3 1 2 3 1 ...
     $ blooms: num  0 0 111 183.5 59.2 ...


The `blooms` variable will be the outcome we want to predict, and `water` and `shade` will be the predictor variables. `water` indicates one of three ordered levels of soil moisture, from low (1) to high (3). `shade` indicates three ordered levels of light exposure, from low (1) to high (3). `bed` indicates a cluster of plants fro the same section of the greenhouse.

We expect `water` and `shade` to help plants grow. *However*, we also expect an interaction between them - increasing the water if there is no light is unlikely to help.

### The models

We'll focus on two models: one with both variables but no interactions, and one with the interactions.

The causal scenario is simple: $W \to B \leftarrow S$. The DAG doesn't tell us about the shape of the function $B = f(S, W)$. In principle, every unique combination of $S$ and $W$ could have its own mean, &c. We'll start simple.

$$
\begin{align*}
B_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_W(W_i - \bar{W}) + \beta_S(S_i - \bar{S})
\end{align*}
$$


```R
d$blooms_std <- d$blooms / max(d$blooms)
d$water_cent <- d$water - mean(d$water)
d$shade_cent <- d$shade - mean(d$shade)
```

Now we need some priors! A vague guess:

$$
\begin{align*}
\alpha &\sim \text{Normal}(0.5, 1) \\
\beta_W &\sim \text{Normal}(0, 1) \\
\beta_S &\sim \text{Normal}(0, 1)
\end{align*}
$$

Centring $\alpha$ at 0.5 implies that if $S$ and $W$ are at their mean values, the blooms will be about half of the maximum size. Centring $\beta_S$ and $\beta_W$ at 0 means that we make no assumptions about the direction of the effect, even though logically we expect that the effect of water should be positive and the effect of shade should be negative.

The prior standard deviations seem too broad. For instance, $\alpha$ should be between 0 and 1. But...


```R
# most of the weight is outside this range!
a <- rnorm(1e4, 0.5, 1)
sum(a < 0 | a > 1) / length(a)
```


0.6164



```R
# changing the std to 0.25 will constrain the mass
a <- rnorm(1e4, 0.5, 0.25)
sum(a < 0 | a > 1) / length(a)
```


0.0473


What about the slopes? The range of both water and shade is 2. In order to go from the minimum (0 blooms) to the maximum (1) in the two units of change would require a slope of 0.5. Assigning a std of 0.25 would make this 2 std from the mean, putting most of the mass in this range.


```R
m8.4 <- quap(
    alist(
        blooms_std ~ dnorm(mu, sigma),
        mu <- a + bw * water_cent + bs * shade_cent,
        a ~ dnorm(0.5, 0.25),
        bw ~ dnorm(0, 0.25),
        bs ~ dnorm(0, 0.25),
        sigma ~ dexp(1)
    ),
    data=d
)
```

Before we actually plot this (&c.), let's build the interaction model. We want the mean for the water slope to depend on the level of shade, *but* we also want the shade slope to depend on the level of water. How can we do this?

Basically, we'll have a linear model inside our linear model. So, to make the slope conditional on another variable, we ust make another linear model in that!

Say $W_i$ and $S_i$ are the centred models. Then we can define the slope $\beta_W$ with its own linear model $\gamma_W$:

$$
\begin{align*}
\mu_i &= \alpha + \gamma_{W,i}W_i + \beta_S S_i \\
\gamma_{W,i} &= \beta_W + \beta_{WS} S_I \\
\end{align*}
$$

Now $\gamma_{W,i}$ is the slope defining how quickly blooms change with water level. The parameter $\beta_W$ is the rate of change, when shade is at its mean value, and $\beta_{WS}$ is the rate of change in $\gamma_{W,i}$ as shade chagnes - the shope for shade on the slope of water.

Now we also want the shade to depend on water. Luckily, we get this for free - once you make $z$ conditional on $x$, $x$ is automatically conditional on $z$.

So now we can just sub this into our equation:

$$
\mu_i = \alpha + \underbrace{( \beta_W + \beta_{WS}S_i )}_{\gamma_{W,i}}W_i + \beta_S S_i = \alpha + \beta_W W_i + \beta_S S_i + \beta_{WS}S_i W_i
$$

That last part is the conventional for form a continuous interaction - a term containing the product of the two variables with its own slope variable.

Our new model:

$$
\begin{align*}
B_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_W W_i + \beta_S S_i + \beta_{WS} W_i S_i
\end{align*}
$$

Now we need a prior! This is a bit tricky, since there isn't really a natural interpretation for $\beta_{WS}$. However, we can work with implied predictions. Say that the strongest effect is one in which enough shade makes the water have zero effect. Then $\gamma_{W,i} = \beta_W + \beta_{WS} S_i = 0$. If we then set $S_i = 1$ (the max), then $\beta_W + \beta_{WS}(1) = 0 \to \beta_{WS} = -\beta_W$.

In code:



```R
m8.5 <- quap(
    alist(
        blooms_std ~ dnorm(mu, sigma),
        mu <- a + bw * water_cent + bs * shade_cent + bws * water_cent * shade_cent,
        a ~ dnorm(0.5, 0.25),
        bw ~ dnorm(0, 0.25),
        bs ~ dnorm(0, 0.25),
        bws ~ dnorm(0, 0.25),
        sigma ~ dexp(1)
    ),
    data=d
)
```

### Plotting posterior predictions

Now that there's an interaction, we can't just plot water and blooms, since the actual effect will depend on the value of shade that we use. Thus, our strategy is to make three plots: one for each level of shade, and plot water vs. blooms on each of these to see what happens.


```R
plot_df <- data.frame(
    water_cent = double(),
    blooms_std = double(),
    shade_cent = integer(),
    group = integer(),
    model = character()
)

# m8.4 information
for (shade in -1:1) {
    mu <- link(m8.4, data=data.frame(shade_cent=shade, water_cent=-1:1))
    for (i in 1:20) {
        new_df <- data.frame(water_cent=-1:1, blooms_std=mu[i,], shade_cent=shade, group=10*shade + i, model="m8.4")
        plot_df <- rbind(plot_df, new_df)
    }
}

# m8.5 information
for (shade in -1:1) {
    mu <- link(m8.5, data=data.frame(shade_cent=shade, water_cent=-1:1))
    for (i in 1:20) {
        new_df <- data.frame(water_cent=-1:1, blooms_std=mu[i,], shade_cent=shade, group=10*shade + i, model="m8.5")
        plot_df <- rbind(plot_df, new_df)
    }
}

options(repr.plot.width=15, repr.plot.height=8)
ggplot() +
    geom_point(data=d, aes(water_cent, blooms_std)) +
    geom_line(data=plot_df, aes(water_cent, blooms_std, group=group), alpha=0.3) +
    facet_grid(vars(model), vars(shade_cent)) +
    xlab("Water") + 
    ylab("Blooms") +
    ggtitle("Posterior")
```


    
![png](sr-ch08-output_50_0.png)
    


In the top one (no interaction), the slope doesn't vary, so it 'believes' that water is always helpful. On the other hand, in the interaction model, it can vary - we can see that water is very helpful when there's no shade, but has very little effect when there is a lot of shade.

### Plotting prior predictions

We can use the same technique to plot the prior predictions!


```R
set.seed(7)
plot_df <- data.frame(
    water_cent = double(),
    blooms_std = double(),
    shade_cent = integer(),
    group = integer(),
    model = character()
)

prior_8.4 <- extract.prior(m8.4)
for (shade in -1:1) {
    mu <- link(m8.4, data=data.frame(shade_cent=shade, water_cent=-1:1), post=prior_8.4)
    for (i in 1:20) {
        new_df <- data.frame(water_cent=-1:1, blooms_std=mu[i,], shade_cent=shade, group=10*shade + i, model="m8.4")
        plot_df <- rbind(plot_df, new_df)
    }
}
prior_8.5 <- extract.prior(m8.5)
for (shade in -1:1) {
    mu <- link(m8.5, data=data.frame(shade_cent=shade, water_cent=-1:1), post=prior_8.5)
    for (i in 1:20) {
        new_df <- data.frame(water_cent=-1:1, blooms_std=mu[i,], shade_cent=shade, group=10*shade + i, model="m8.5")
        plot_df <- rbind(plot_df, new_df)
    }
}

ggplot() +
    geom_point(data=d, aes(water_cent, blooms_std)) +
    geom_line(data=plot_df, aes(water_cent, blooms_std, group=group), alpha=0.3) +
    facet_grid(vars(model), vars(shade_cent)) +
    xlab("Water") + 
    ylab("Blooms") +
    ggtitle("Prior")
```


    
![png](sr-ch08-output_52_0.png)
    


Not such great priors, as we expected. Probably the best you can say is that they don't bias the effects to either positive or negative and they weakly constrain the results to be plausible.

## Practice

#### 8E1
For each of the causal relationships below, name a hypothetical third variable that would lead to an interaction effect
1. Bread dough rises because of yeast
1. Education leads to higher income
1. Gasoline makes the car go

1. A third variable could be the temperature: too low and the yeast will never rise (no matter how much there is), and the same with if it's too hot.
1. An interaction variable could be the GDP of the country you're in - if it is very low, then your chances of having a higher income are low too, no matter how well educated you are.
1. An interaction variable could be whether the brakes are on - if they are, then no amount of gas will make it go!

#### 8E2
Which of the following explanations invokes an interaction?
1. Caramelizing onions requires cooking over low heat and making sure the onions do not dry out
1. A car will go faster when it has mor cylinders or when it has a better fuel injector
1. Most people get their political beliefs from their parents, unless they get them instead from their friends
1. Intelligent animal species tend to be either highly social or have minupulative appendages (hands, tentacles, &c.)

1. Yes - requires the interaction of low heat and moisture
1. No - either one will make it go faster, regardless of the other
1. No - their beliefs are coming from either / or, not the interaction between them
1. No - Again, these are independent.

#### 8E3
for each of the relationships above, write a linear model that expresses the stated relationship

1. $c = \alpha + \beta_h * h + \beta_m * m + \beta_{hm} * m * h$, where $h$ is the temperature and $m$ is the moisture level.
1. $s = \alpha + \beta_c * c + \beta_f * f$, where $c$ is the number of cylinders and $f$ is a measure of the quality of the fuel injectors
1. $b = \alpha + \beta_p * p + \beta_f * f$, where $p$ is some measure of your parents' political beliegs and $f$ is a measure of your friends' beliefs
1. $I = \alpha + \beta_S * S + \beta_M * M$, where $S$ is a measure of sociality and $M$ is a measure of manipulativeness of appendages.

#### 8M1
Recall the tulips example from the chapter. Suppose another set of treatment adjusted the temperature in the greenhouse over two levels: cold and hot. The data in the chapter were collected at the cold temperature. You find that none of the plant growin under the hot temperature developed any blooms at all, regardless of the water and shade levels. Can you explain this result in terms of interactions between water, shade, and temperature?

Yes - tulips are a cool-season plant. Just like there is an interaction between light and water, there is one with temperature - if it is too high or too low, then no amount of the others will make a difference.

#### 8M2
Can you invent a regression equation that would make the bloom size zero, whenever the temperature is hot?

Yes - we could multiply the entire model by an indicator variable of whether the temperature is hot ($H=1$) or cold ($H=0$). So something like

$$
b = H * (\text{rest of the model here})
$$

#### 8M3
In parts of North America, ravens depend upon wolves for their food. This is because ravens are carnivorous, but cannot usually kill or open carcasses of prey. Wolves can and do kill and tear open animals, and they tolerate ravens co-feeding at theis kills. This species relationship is generally descibed as a "species interaction". Can you invent a hypothetical set of data on raven population size in which this relationsup would manifest as a statistical interaction? Do you think that biological interaction would be linear? Why or why not?

Let's assume that the raven population is dependent on both the raw amounts of food and the amount of wolves. This means that if there is a lot of food, ravens will generally do OK even if there are no wolves, but adding in the wolves will increase the amount that is available for them to eat. Then our model will look something like as follows:

$$
\begin{align*}
R_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_F * F_i + \beta_W * W_i + \beta_{FW}*F_i * W_i \\
\end{align*}
$$

where $R$ is the raven population, $F$ is the general availability of food, and $W$ is the size of the wolf population. We can simulate this data:


```R
NUM_SAMPLES <- 30
SIGMA = 0.2
F <- seq(from=0, to=1, length.out=NUM_SAMPLES)
W <- seq(from=0, to=1, length.out=NUM_SAMPLES)
R <- F + W + F * W + rnorm(NUM_SAMPLES, 0, 1)
```


```R
# now to test it
ravens_data <- data.frame(
    F=F,
    W=W,
    R=R
)
m8m3 <- quap(
    alist(
        R ~ dnorm(mu, sigma),
        mu <- a + bF * F + bW * W + bWF * W * F,
        a ~ dnorm(0, 0.25),
        bF ~ dnorm(0, 0.25),
        bW ~ dnorm(0, 0.25),
        bWF ~ dnorm(0, 0.25),
        sigma ~ dexp(1)
    ),
    data=ravens_data
)
plot(precis(m8m3))
```


    
![png](sr-ch08-output_68_0.png)
    


So yes, this shows an interaction effect. All else being equal, I can imagine this being a linear relationship - if there are twice as many wolves then each of those would be taking a proportionaly larger amount of prey, and so that would make more available to the ravens.

#### 8M4
Repeat the tulips analysis, but now use priors that constrain the effect of water to be positive and the effect of shade to be negative. Use prior predictive simulation. What do these prior assumptions mean for the interaction prior, if anything?

For the interaction prior: once again, we'll go with the idea that the strongest effect is one which makes water have zero effect. Then

$$
\beta_w + \beta_{WS}S_I = 0
$$

Again, setting $S_I=1$ (strongest shade), we get $\beta_{ WS } = -\beta_w$. Since our prior mean for $\beta_w = 0.5$, we want $\beta_{WS} = -0.5$.


```R
m8m4 <- quap(
    alist(
        blooms_std ~ dnorm(mu, sigma),
        mu <- a + bw * water_cent + bs * shade_cent + bws * water_cent * shade_cent,
        a ~ dnorm(0.5, 0.25),
        bw ~ dnorm(0.5, 0.25),
        bs ~ dnorm(-0.5, 0.25),
        bws ~ dnorm(-0.5, 0.25),
        sigma ~ dexp(1)
    ),
    data=d
)
```


```R
plot_df <- data.frame(water_cent=double(), blooms_std=double(), shade_cent=double(), group=integer())
prior_m8m4 <- extract.prior(m8m4)
for (shade in -1:1) {
    mu <- link(m8m4, data=data.frame(shade_cent=shade, water_cent=-1:1), post=prior_m8m4)
    for (i in 1:20) {
        new_df <- data.frame(water_cent=-1:1, blooms_std=mu[i,], shade_cent=shade, group=10*shade + i)
        plot_df <- rbind(plot_df, new_df)
    }
}

ggplot() +
    geom_point(data=d, aes(water_cent, blooms_std)) +
    geom_line(data=plot_df, aes(water_cent, blooms_std, group=group), alpha=0.3) +
    facet_grid(vars(shade_cent)) +
    xlab("Water") + 
    ylab("Blooms") +
    ggtitle("Prior")

## Posterior

plot_df <- data.frame(water_cent=double(), blooms_std=double(), shade_cent=double(), group=integer())
for (shade in -1:1) {
    mu <- link(m8m4, data=data.frame(shade_cent=shade, water_cent=-1:1))
    for (i in 1:20) {
        new_df <- data.frame(water_cent=-1:1, blooms_std=mu[i,], shade_cent=shade, group=10*shade + i)
        plot_df <- rbind(plot_df, new_df)
    }
}

ggplot() +
    geom_point(data=d, aes(water_cent, blooms_std)) +
    geom_line(data=plot_df, aes(water_cent, blooms_std, group=group), alpha=0.3) +
    facet_grid(vars(shade_cent)) +
    xlab("Water") + 
    ylab("Blooms") +
    ggtitle("Posterior")
```


    
![png](sr-ch08-output_73_0.png)
    



    
![png](sr-ch08-output_73_1.png)
    



```R
precis(m8m4)
```


<table class="dataframe">
<caption>A precis: 5 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td> 0.3579862</td><td>0.02394950</td><td> 0.31971024</td><td> 0.39626210</td></tr>
	<tr><th scope=row>bw</th><td> 0.2135758</td><td>0.02928128</td><td> 0.16677870</td><td> 0.26037300</td></tr>
	<tr><th scope=row>bs</th><td>-0.1203097</td><td>0.02929603</td><td>-0.16713044</td><td>-0.07348902</td></tr>
	<tr><th scope=row>bws</th><td>-0.1533638</td><td>0.03576801</td><td>-0.21052797</td><td>-0.09619960</td></tr>
	<tr><th scope=row>sigma</th><td> 0.1250064</td><td>0.01700571</td><td> 0.09782803</td><td> 0.15218485</td></tr>
</tbody>
</table>



This has a slight effect on the interaction parameter.

#### 8H1

Return to the `data(tulips)` example in the chapter. Now include the `bed` variable as a predictor in the interaction model. Don't interact `bed` with the other predictors; just include it as a main effect. Note that `bed` is categorical, so to use it properly you'll need to either construct dummy variables or an index variable, as described in chapter 5.


```R
d$bed
d$bed_num <- ifelse(d$bed == 'a', 1, d$bed)
d$bed_num <- ifelse(d$bed == 'b', 2, d$bed)
d$bed_num <- ifelse(d$bed == 'c', 3, d$bed)
d$bed_num
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>a</li><li>a</li><li>a</li><li>a</li><li>a</li><li>a</li><li>a</li><li>a</li><li>a</li><li>b</li><li>b</li><li>b</li><li>b</li><li>b</li><li>b</li><li>b</li><li>b</li><li>b</li><li>c</li><li>c</li><li>c</li><li>c</li><li>c</li><li>c</li><li>c</li><li>c</li><li>c</li></ol>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<style>
	.list-inline {list-style: none; margin:0; padding: 0}
	.list-inline>li {display: inline-block}
	.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
	</style>
	<ol class=list-inline><li>'a'</li><li>'b'</li><li>'c'</li></ol>
</details>



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>1</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>2</li><li>3</li><li>3</li><li>3</li><li>3</li><li>3</li><li>3</li><li>3</li><li>3</li><li>3</li></ol>




```R
m8h1 <- quap(
    alist(
        blooms_std ~ dnorm(mu, sigma),
        mu <- a + bw * water_cent + bs * shade_cent + bb[bed_num] * bed_num + bws * water_cent * shade_cent,
        a ~ dnorm(0.5, 0.25),
        bw ~ dnorm(0, 0.25),
        bs ~ dnorm(0, 0.25),
        bb[bed_num] ~ dnorm(0, 0.25),
        bws ~ dnorm(0, 0.25),
        sigma ~ dexp(1)
    ),
    data=d
)
precis(m8h1, depth=2)
```


<table class="dataframe">
<caption>A precis: 8 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td> 0.387428428</td><td>0.16345856</td><td> 0.12619008</td><td> 0.64866677</td></tr>
	<tr><th scope=row>bw</th><td> 0.207437069</td><td>0.02536622</td><td> 0.16689695</td><td> 0.24797719</td></tr>
	<tr><th scope=row>bs</th><td>-0.113848869</td><td>0.02536160</td><td>-0.15438160</td><td>-0.07331614</td></tr>
	<tr><th scope=row>bb[1]</th><td>-0.116455837</td><td>0.16405627</td><td>-0.37864945</td><td> 0.14573777</td></tr>
	<tr><th scope=row>bb[2]</th><td> 0.003389892</td><td>0.08327055</td><td>-0.12969253</td><td> 0.13647232</td></tr>
	<tr><th scope=row>bb[3]</th><td> 0.006582216</td><td>0.05566993</td><td>-0.08238908</td><td> 0.09555351</td></tr>
	<tr><th scope=row>bws</th><td>-0.143893326</td><td>0.03098525</td><td>-0.19341375</td><td>-0.09437291</td></tr>
	<tr><th scope=row>sigma</th><td> 0.108149529</td><td>0.01467928</td><td> 0.08468920</td><td> 0.13160985</td></tr>
</tbody>
</table>



For all of the `bb` coefficients, they seem to hover around 0, indicating to us that maybe the bed doesn't have much of an effect on the number of blooms.

#### 8H2
Use WAIC to compare the model from 8H1 to a model that omits `bed`. What do you infer from this comparison? can you reconcile the WAIC results with the posterior distribution of the `bed` coefficients?


```R
compare(m8.5, m8h1)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m8h1</th><td>-22.56145</td><td>10.24801</td><td>0.000000</td><td>      NA</td><td>10.225422</td><td>0.6465465</td></tr>
	<tr><th scope=row>m8.5</th><td>-21.35366</td><td>10.55084</td><td>1.207786</td><td>8.439126</td><td> 6.930282</td><td>0.3534535</td></tr>
</tbody>
</table>



Here we see that the `m8h1` model is slightly better (1.6), but the `dSE` is 7.7, so it's not appreciably better. The models are roughly equivalent,which is in line with the finding that the bed number doesn't really affect the model too much.

#### 8H3
Consider, again, the `data(rugged)` data on evonomic development and terrain ruggedness. On of the African countries in that example, Seychelles, is far outside the cloud of other nations, being a rare country with both relatively high GDP and high ruggedness. Seychelles is also unusual in that it is a group of islands far from the coast of mainland Africa, and its main economic activity is tourism.

a) Focus on model m8.5  [NB this should be 8.3 - 8.5 is the tulip one] from this chapter. Use WAIC pointwise penalties and PSIS Pareto-$k$ values to measure relative incluence of each country. By these criteria, is Seychelles influencing the results? Are there other nations that are relatively influential? If so, can you explain why?

b) Now use robust regression, as described in the previous chapter. Modify `m8.5` [NB: `m8.3`] to use a Student-t distribution with $\nu=2$. Does this change the results in a substantial way?


```R
# comparison_data$country <- dd$country
waic_data <- WAIC(m8.3, pointwise=T)
psis_data <- PSIS(m8.3, pointwise=T)

comparison_data <- cbind(dd$country, waic_data, psis_data)

comparison_data
```

    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    



<table class="dataframe">
<caption>A data.frame: 170 × 10</caption>
<thead>
	<tr><th scope=col>dd$country</th><th scope=col>WAIC</th><th scope=col>lppd</th><th scope=col>penalty</th><th scope=col>std_err</th><th scope=col>PSIS</th><th scope=col>lppd</th><th scope=col>penalty</th><th scope=col>std_err</th><th scope=col>k</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Angola                  </td><td>-2.5621804</td><td> 1.2841356</td><td>0.003045424</td><td>15.17878</td><td>-2.5678652</td><td> 1.2839326</td><td>0.002982210</td><td>15.31781</td><td>-0.0904058847</td></tr>
	<tr><td>Albania                 </td><td>-2.4221347</td><td> 1.2183020</td><td>0.007234686</td><td>15.17878</td><td>-2.4332127</td><td> 1.2166063</td><td>0.006803377</td><td>15.31781</td><td> 0.1164936058</td></tr>
	<tr><td>United Arab Emirates    </td><td>-1.6801218</td><td> 0.8493764</td><td>0.009315510</td><td>15.17878</td><td>-1.6777420</td><td> 0.8388710</td><td>0.009292557</td><td>15.31781</td><td> 0.1684758923</td></tr>
	<tr><td>Argentina               </td><td>-2.4307238</td><td> 1.2191537</td><td>0.003791758</td><td>15.17878</td><td>-2.4333680</td><td> 1.2166840</td><td>0.003648478</td><td>15.31781</td><td> 0.1019437353</td></tr>
	<tr><td>Armenia                 </td><td>-1.6254743</td><td> 0.8307847</td><td>0.018047533</td><td>15.17878</td><td>-1.6431173</td><td> 0.8215586</td><td>0.017573559</td><td>15.31781</td><td> 0.0172241645</td></tr>
	<tr><td>Antigua and Barbuda     </td><td>-2.5637113</td><td> 1.2849934</td><td>0.003137730</td><td>15.17878</td><td>-2.5671393</td><td> 1.2835696</td><td>0.003109582</td><td>15.31781</td><td>-0.0755641782</td></tr>
	<tr><td>Australia               </td><td>-1.4746336</td><td> 0.7562978</td><td>0.018981017</td><td>15.17878</td><td>-1.4737432</td><td> 0.7368716</td><td>0.020252918</td><td>15.31781</td><td> 0.1240474677</td></tr>
	<tr><td>Austria                 </td><td> 1.1603749</td><td>-0.4220944</td><td>0.158092987</td><td>15.17878</td><td> 1.2198133</td><td>-0.6099067</td><td>0.160659992</td><td>15.31781</td><td> 0.1358448151</td></tr>
	<tr><td>Azerbaijan              </td><td>-1.3367322</td><td> 0.6795020</td><td>0.011135938</td><td>15.17878</td><td>-1.3480917</td><td> 0.6740459</td><td>0.010375584</td><td>15.31781</td><td> 0.0334377941</td></tr>
	<tr><td>Burundi                 </td><td>-0.8611712</td><td> 0.4753738</td><td>0.044788263</td><td>15.17878</td><td>-0.8450945</td><td> 0.4225473</td><td>0.042514686</td><td>15.31781</td><td>-0.0261737736</td></tr>
	<tr><td>Belgium                 </td><td>-1.1954013</td><td> 0.6171697</td><td>0.019469047</td><td>15.17878</td><td>-1.1918938</td><td> 0.5959469</td><td>0.020606093</td><td>15.31781</td><td> 0.0645048920</td></tr>
	<tr><td>Benin                   </td><td>-2.2943064</td><td> 1.1567927</td><td>0.009639543</td><td>15.17878</td><td>-2.2903163</td><td> 1.1451581</td><td>0.010040517</td><td>15.31781</td><td>-0.1054806192</td></tr>
	<tr><td>Burkina Faso            </td><td>-2.3217487</td><td> 1.1693582</td><td>0.008483819</td><td>15.17878</td><td>-2.3183965</td><td> 1.1591982</td><td>0.008812219</td><td>15.31781</td><td>-0.1011696874</td></tr>
	<tr><td>Bangladesh              </td><td> 1.5970698</td><td>-0.6946261</td><td>0.103908745</td><td>15.17878</td><td> 1.6100285</td><td>-0.8050142</td><td>0.101981869</td><td>15.31781</td><td> 0.0114453380</td></tr>
	<tr><td>Bulgaria                </td><td>-2.5174145</td><td> 1.2617767</td><td>0.003069460</td><td>15.17878</td><td>-2.5239295</td><td> 1.2619647</td><td>0.003026163</td><td>15.31781</td><td>-0.0003086785</td></tr>
	<tr><td>Bahrain                 </td><td>-2.2585342</td><td> 1.1359738</td><td>0.006706687</td><td>15.17878</td><td>-2.2613015</td><td> 1.1306508</td><td>0.006770829</td><td>15.31781</td><td> 0.0275545617</td></tr>
	<tr><td>Bahamas                 </td><td>-2.2180639</td><td> 1.1171795</td><td>0.008147590</td><td>15.17878</td><td>-2.2210144</td><td> 1.1105072</td><td>0.008377612</td><td>15.31781</td><td> 0.0254123781</td></tr>
	<tr><td>Bosnia and Herzegovina  </td><td>-2.5305339</td><td> 1.2685949</td><td>0.003327955</td><td>15.17878</td><td>-2.5378850</td><td> 1.2689425</td><td>0.003184340</td><td>15.31781</td><td> 0.0407853770</td></tr>
	<tr><td>Belarus                 </td><td>-1.9936814</td><td> 1.0073675</td><td>0.010526784</td><td>15.17878</td><td>-1.9937113</td><td> 0.9968556</td><td>0.010849049</td><td>15.31781</td><td> 0.0950596095</td></tr>
	<tr><td>Belize                  </td><td>-2.3743270</td><td> 1.1913325</td><td>0.004169000</td><td>15.17878</td><td>-2.3784491</td><td> 1.1892245</td><td>0.004276458</td><td>15.31781</td><td>-0.2147702595</td></tr>
	<tr><td>Bolivia                 </td><td>-0.7201161</td><td> 0.3812678</td><td>0.021209746</td><td>15.17878</td><td>-0.7230895</td><td> 0.3615448</td><td>0.020294193</td><td>15.31781</td><td>-0.0241378206</td></tr>
	<tr><td>Brazil                  </td><td>-2.4855154</td><td> 1.2465684</td><td>0.003810695</td><td>15.17878</td><td>-2.4885695</td><td> 1.2442848</td><td>0.003934526</td><td>15.31781</td><td>-0.1322116554</td></tr>
	<tr><td>Barbados                </td><td>-2.1352413</td><td> 1.0726928</td><td>0.005072103</td><td>15.17878</td><td>-2.1350211</td><td> 1.0675105</td><td>0.004860210</td><td>15.31781</td><td> 0.0902473820</td></tr>
	<tr><td>Botswana                </td><td> 0.5518793</td><td>-0.1718791</td><td>0.104060519</td><td>15.17878</td><td> 0.5342061</td><td>-0.2671030</td><td>0.104401691</td><td>15.31781</td><td> 0.1757067283</td></tr>
	<tr><td>Central African Republic</td><td>-2.4496649</td><td> 1.2303797</td><td>0.005547244</td><td>15.17878</td><td>-2.4492198</td><td> 1.2246099</td><td>0.005816214</td><td>15.31781</td><td>-0.0938024515</td></tr>
	<tr><td>Canada                  </td><td>-1.0086045</td><td> 0.5210337</td><td>0.016731418</td><td>15.17878</td><td>-1.0019722</td><td> 0.5009861</td><td>0.017126111</td><td>15.31781</td><td> 0.0430510753</td></tr>
	<tr><td>Switzerland             </td><td> 2.7677389</td><td>-0.9338118</td><td>0.450057662</td><td>15.17878</td><td> 2.8719520</td><td>-1.4359760</td><td>0.463453971</td><td>15.31781</td><td> 0.1459685514</td></tr>
	<tr><td>Chile                   </td><td>-2.3882547</td><td> 1.1989639</td><td>0.004836478</td><td>15.17878</td><td>-2.3848883</td><td> 1.1924441</td><td>0.005084131</td><td>15.31781</td><td>-0.0180728484</td></tr>
	<tr><td>China                   </td><td>-2.2002465</td><td> 1.1049700</td><td>0.004846732</td><td>15.17878</td><td>-2.2106993</td><td> 1.1053497</td><td>0.004498033</td><td>15.31781</td><td>-0.0194472755</td></tr>
	<tr><td>Cote d'Ivoire           </td><td>-2.5544941</td><td> 1.2803884</td><td>0.003141331</td><td>15.17878</td><td>-2.5598881</td><td> 1.2799440</td><td>0.003023829</td><td>15.31781</td><td>-0.0617363212</td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>Slovakia                          </td><td>-2.3624299</td><td> 1.18486727</td><td>0.003652315</td><td>15.17878</td><td>-2.3621279</td><td> 1.18106397</td><td>0.003555856</td><td>15.31781</td><td>-0.101967757</td></tr>
	<tr><td>Slovenia                          </td><td>-1.3681505</td><td> 0.70379026</td><td>0.019715013</td><td>15.17878</td><td>-1.3498571</td><td> 0.67492856</td><td>0.019864553</td><td>15.31781</td><td> 0.022015961</td></tr>
	<tr><td>Sweden                            </td><td>-1.1761459</td><td> 0.60334077</td><td>0.015267838</td><td>15.17878</td><td>-1.1708866</td><td> 0.58544331</td><td>0.015664989</td><td>15.31781</td><td> 0.050264134</td></tr>
	<tr><td>Swaziland                         </td><td>-2.0817177</td><td> 1.07132741</td><td>0.030468542</td><td>15.17878</td><td>-2.0905738</td><td> 1.04528692</td><td>0.030811246</td><td>15.31781</td><td> 0.182094508</td></tr>
	<tr><td>Seychelles                        </td><td> 1.3201405</td><td>-0.04809972</td><td>0.611970551</td><td>15.17878</td><td> 1.4342282</td><td>-0.71711409</td><td>0.671448239</td><td>15.31781</td><td> 0.542272781</td></tr>
	<tr><td>Syrian Arab Republic              </td><td>-1.4599382</td><td> 0.74203002</td><td>0.012060941</td><td>15.17878</td><td>-1.4632500</td><td> 0.73162500</td><td>0.011770032</td><td>15.31781</td><td>-0.044725403</td></tr>
	<tr><td>Chad                              </td><td>-2.0242749</td><td> 1.02635344</td><td>0.014216003</td><td>15.17878</td><td>-2.0165358</td><td> 1.00826788</td><td>0.014276063</td><td>15.31781</td><td> 0.023180629</td></tr>
	<tr><td>Togo                              </td><td>-2.5468885</td><td> 1.27672388</td><td>0.003279615</td><td>15.17878</td><td>-2.5504858</td><td> 1.27524290</td><td>0.003305570</td><td>15.31781</td><td>-0.033370554</td></tr>
	<tr><td>Thailand                          </td><td>-2.4977323</td><td> 1.25200901</td><td>0.003142881</td><td>15.17878</td><td>-2.5033767</td><td> 1.25168834</td><td>0.003158290</td><td>15.31781</td><td>-0.063619023</td></tr>
	<tr><td>Tajikistan                        </td><td> 0.4906473</td><td> 0.04793939</td><td>0.293263050</td><td>15.17878</td><td> 0.4755517</td><td>-0.23777585</td><td>0.315545996</td><td>15.31781</td><td> 0.331926920</td></tr>
	<tr><td>Turkmenistan                      </td><td>-1.5205168</td><td> 0.77695203</td><td>0.016693611</td><td>15.17878</td><td>-1.5195363</td><td> 0.75976814</td><td>0.016856358</td><td>15.31781</td><td> 0.034530476</td></tr>
	<tr><td>Tonga                             </td><td>-2.4905258</td><td> 1.24855577</td><td>0.003292879</td><td>15.17878</td><td>-2.4953632</td><td> 1.24768161</td><td>0.003350500</td><td>15.31781</td><td>-0.139502782</td></tr>
	<tr><td>Trinidad and Tobago               </td><td>-2.5718340</td><td> 1.28892790</td><td>0.003010881</td><td>15.17878</td><td>-2.5759926</td><td> 1.28799629</td><td>0.002982841</td><td>15.31781</td><td>-0.114586954</td></tr>
	<tr><td>Tunisia                           </td><td>-0.5534190</td><td> 0.32244696</td><td>0.045737476</td><td>15.17878</td><td>-0.5709465</td><td> 0.28547323</td><td>0.045029938</td><td>15.31781</td><td> 0.132712858</td></tr>
	<tr><td>Turkey                            </td><td>-2.5579508</td><td> 1.28207173</td><td>0.003096318</td><td>15.17878</td><td>-2.5605656</td><td> 1.28028279</td><td>0.003164918</td><td>15.31781</td><td>-0.139145387</td></tr>
	<tr><td>United Republic of Tanzania       </td><td>-0.9370206</td><td> 0.50561774</td><td>0.037107456</td><td>15.17878</td><td>-0.9193232</td><td> 0.45966159</td><td>0.035321888</td><td>15.31781</td><td> 0.145467520</td></tr>
	<tr><td>Uganda                            </td><td>-2.4230281</td><td> 1.21657060</td><td>0.005056525</td><td>15.17878</td><td>-2.4223056</td><td> 1.21115279</td><td>0.005129773</td><td>15.31781</td><td> 0.006088392</td></tr>
	<tr><td>Ukraine                           </td><td>-1.8103750</td><td> 0.91604226</td><td>0.010854771</td><td>15.17878</td><td>-1.8116260</td><td> 0.90581302</td><td>0.010983562</td><td>15.31781</td><td> 0.060678796</td></tr>
	<tr><td>Uruguay                           </td><td>-2.5680619</td><td> 1.28705669</td><td>0.003025713</td><td>15.17878</td><td>-2.5720669</td><td> 1.28603344</td><td>0.003018678</td><td>15.31781</td><td>-0.045938221</td></tr>
	<tr><td>United States of America          </td><td>-0.1701940</td><td> 0.11104023</td><td>0.025943223</td><td>15.17878</td><td>-0.1560816</td><td> 0.07804079</td><td>0.026169024</td><td>15.31781</td><td> 0.049111505</td></tr>
	<tr><td>Uzbekistan                        </td><td> 1.1766485</td><td>-0.51570905</td><td>0.072615193</td><td>15.17878</td><td> 1.1827404</td><td>-0.59137018</td><td>0.070434407</td><td>15.31781</td><td>-0.015889114</td></tr>
	<tr><td>Saint Vincent and the Grenadines  </td><td>-2.5562636</td><td> 1.28134093</td><td>0.003209147</td><td>15.17878</td><td>-2.5606385</td><td> 1.28031925</td><td>0.003158821</td><td>15.31781</td><td>-0.066441324</td></tr>
	<tr><td>Venezuela (Bolivarian Republic of)</td><td>-2.3482243</td><td> 1.17841156</td><td>0.004299422</td><td>15.17878</td><td>-2.3524200</td><td> 1.17620999</td><td>0.004402447</td><td>15.31781</td><td>-0.196561158</td></tr>
	<tr><td>Viet Nam                          </td><td>-0.8825663</td><td> 0.46152604</td><td>0.020242888</td><td>15.17878</td><td>-0.8976649</td><td> 0.44883243</td><td>0.019404347</td><td>15.31781</td><td> 0.209810873</td></tr>
	<tr><td>Vanuatu                           </td><td>-1.8119859</td><td> 0.91306208</td><td>0.007069130</td><td>15.17878</td><td>-1.8230585</td><td> 0.91152927</td><td>0.006545006</td><td>15.31781</td><td>-0.044266245</td></tr>
	<tr><td>Samoa                             </td><td>-2.3958266</td><td> 1.20145704</td><td>0.003543757</td><td>15.17878</td><td>-2.4041424</td><td> 1.20207118</td><td>0.003378832</td><td>15.31781</td><td> 0.000514957</td></tr>
	<tr><td>Yemen                             </td><td> 2.5968168</td><td>-1.17902770</td><td>0.119380708</td><td>15.17878</td><td> 2.5830047</td><td>-1.29150233</td><td>0.121329260</td><td>15.31781</td><td> 0.356410718</td></tr>
	<tr><td>South Africa                      </td><td> 0.2144980</td><td>-0.03104359</td><td>0.076205420</td><td>15.17878</td><td> 0.2014892</td><td>-0.10074460</td><td>0.077178292</td><td>15.31781</td><td> 0.180620088</td></tr>
	<tr><td>Zambia                            </td><td>-1.8927005</td><td> 0.96253029</td><td>0.016180038</td><td>15.17878</td><td>-1.8835616</td><td> 0.94178082</td><td>0.015995535</td><td>15.31781</td><td> 0.091639854</td></tr>
	<tr><td>Zimbabwe                          </td><td>-2.4569400</td><td> 1.23309695</td><td>0.004626937</td><td>15.17878</td><td>-2.4663891</td><td> 1.23319457</td><td>0.004340571</td><td>15.31781</td><td>-0.065715287</td></tr>
</tbody>
</table>




```R
# now we want the one with a Pareto-k value which is too high
comparison_data[comparison_data$k > 0.5,]
```


<table class="dataframe">
<caption>A data.frame: 2 × 10</caption>
<thead>
	<tr><th></th><th scope=col>dd$country</th><th scope=col>WAIC</th><th scope=col>lppd</th><th scope=col>penalty</th><th scope=col>std_err</th><th scope=col>PSIS</th><th scope=col>lppd</th><th scope=col>penalty</th><th scope=col>std_err</th><th scope=col>k</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>93</th><td>Lesotho   </td><td>-1.253275</td><td> 0.89670237</td><td>0.2700650</td><td>15.17878</td><td>-1.101587</td><td> 0.5507933</td><td>0.3437452</td><td>15.31781</td><td>0.6829880</td></tr>
	<tr><th scope=row>145</th><td>Seychelles</td><td> 1.320141</td><td>-0.04809972</td><td>0.6119706</td><td>15.17878</td><td> 1.434228</td><td>-0.7171141</td><td>0.6714482</td><td>15.31781</td><td>0.5422728</td></tr>
</tbody>
</table>



So it looks like both Lesotho and the Seychelles are outliers here. Lesotho is possibly influencing it so much simply because its GDP is not all that unusual, but its ruggedness is? Unclear.


```R
m8.3_robust <- quap(
    alist(
        log_gdp_std ~ dstudent(2, mu, sigma),
        mu <- a[cid] + b[cid] * (rugged_std - 0.215),
        a[cid] ~ dnorm(1, 0.1),
        b[cid] ~ dnorm(0, 0.3),
        sigma ~ dexp(1)
    ),
    data=dd
)
```


```R
plot(precis(m8.3, depth=2))
plot(precis(m8.3_robust, depth=2))
```


    
![png](sr-ch08-output_88_0.png)
    



    
![png](sr-ch08-output_88_1.png)
    



```R
coeftab(m8.3, m8.3_robust)
```


          m8.3    m8.3_robust
    a[1]     0.89    0.86    
    a[2]     1.05    1.05    
    b[1]     0.13    0.11    
    b[2]    -0.14   -0.21    
    sigma    0.11    0.08    
    nobs      170     170    


So no, this doesn't seem to have affected the results very much at all.

#### 8H4
The values in `data(nettle)` are data on language diversity in 74 nations. 
- `country`: name of the country
- `num.lang`: Number of recognized languages spoken
- `area`: area in square kilometres
- `k.pop`: population in thousands
- `num.stations`: number of weather stations the provide data for the next two columns
- `mean.growing.season`: average length of the growing season, in months
- `sd.growing.season`: standard deviation of length of growing season, in months

Use these data to evaluate the hypothesis that language diversity is partly a product of food security. The idea is that if there is lots of food, then social groups can be smaller and more self-sufficient, leading to more languages. Use number of languages per capita as the outcome:

```R
d$lang.per.cap <- d$num.lang / d$k.pop
```

Use the log of this as the regression outcome. Very open-ended question. Just try to honestly evaluate the effect of both `mean.growing.season` and `sd.growing.season`, as well as they two-way interaction. Three things to help:

a) evaluate the hypothesis that language diversity (`log(lang.per.cap)`) is positively associate with the average length of the growing season, `mean.growing.season`. Consider `log(area)` as a covariate (not an interaction). Interpret your results.

b) Now evaluate the hypothesis that language diversity is negatively ssociated with the standard deviation of length of growing season, `sm.growing.season`. The idea here is that uncertainty in growing season will lead to insurance in the form of larger social groups. Again, condier `log(area)` as a covariate, not an interaction. Interpret your results.

c) Finally, evaluate the hypothesis that `mean.growing.season` and `sd.growing.season` interact to synergistically reduce number of languages. The idea is that, in nations with longer average growing seasons, high variance makes storage and redistrivution even more important that it would e otherwise. that way, people can cooperate to preserve and protect windfalls to be used during the droghts.


```R
data(nettle)
d <- nettle
d$lang.per.cap <- d$num.lang / d$k.pop
d$log.lang.per.cap <- log(d$lang.per.cap)
d$std.log.lang.per.cap <- d$log.lang.per.cap / max(d$log.lang.per.cap)
d$log.area <- log(d$area)
d$std.log.area <- d$log.area / max(d$log.area)
d$std.growing.season <- d$mean.growing.season / max(d$mean.growing.season)
d$std.sd.growing.season <- d$sd.growing.season / max(d$sd.growing.season)

```


```R
m8h4.a <- quap(
    alist(
        std.log.lang.per.cap ~ dnorm(mu, sigma),
        mu <- a + bA * std.log.area + bG * std.growing.season,
        a ~ dnorm(0, 0.5),
        bA ~ dnorm(0, 0.25),
        bG ~ dnorm(0, 0.25),
        sigma ~ dexp(1)
    ),
    data=d
)
precis(m8h4.a)
plot(precis(m8h4.a))
```


<table class="dataframe">
<caption>A precis: 4 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td> 1.5921720</td><td>0.5248311</td><td> 0.75339055</td><td> 2.4309535</td></tr>
	<tr><th scope=row>bA</th><td> 0.3254743</td><td>0.2521447</td><td>-0.07750167</td><td> 0.7284503</td></tr>
	<tr><th scope=row>bG</th><td> 0.2211754</td><td>0.2505414</td><td>-0.17923816</td><td> 0.6215890</td></tr>
	<tr><th scope=row>sigma</th><td>11.9139103</td><td>1.0056668</td><td>10.30666049</td><td>13.5211601</td></tr>
</tbody>
</table>




    
![png](sr-ch08-output_93_1.png)
    


So it looks like both the area and the length of the growing season are not really associated with the log number of languages per capita.



```R
m8h4.b <- quap(
    alist(
        std.log.lang.per.cap ~ dnorm(mu, sigma),
        mu <- a + bA * std.log.area + bS * std.sd.growing.season,
        a ~ dnorm(0, 0.5),
        bA ~ dnorm(0, 0.25),
        bS ~ dnorm(0, 0.25),
        sigma ~ dexp(1)
    ),
    data=d
)
precis(m8h4.b)
plot(precis(m8h4.b))
```


<table class="dataframe">
<caption>A precis: 4 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td> 1.5845071</td><td>0.5245405</td><td> 0.74619012</td><td> 2.4228241</td></tr>
	<tr><th scope=row>bA</th><td> 0.3240114</td><td>0.2521144</td><td>-0.07891616</td><td> 0.7269389</td></tr>
	<tr><th scope=row>bS</th><td> 0.1203791</td><td>0.2501955</td><td>-0.27948166</td><td> 0.5202398</td></tr>
	<tr><th scope=row>sigma</th><td>11.9915718</td><td>1.0044622</td><td>10.38624727</td><td>13.5968964</td></tr>
</tbody>
</table>




    
![png](sr-ch08-output_95_1.png)
    


Again, we see that the area and the sd of the growing season are not really correlated with the number of languages per capita.


```R
m8h4.c <- quap(
    alist(
        std.log.lang.per.cap ~ dnorm(mu, sigma),
        mu <- a + bA * std.log.area + bG * std.growing.season + bS * std.sd.growing.season + bGS * std.growing.season * std.sd.growing.season,
        a ~ dnorm(0, 0.5),
        bA ~ dnorm(0, 0.25),
        bS ~ dnorm(0, 0.25),
        bG ~ dnorm(0, 0.25),
        bGS ~ dnorm(0, 0.25),
        sigma ~ dexp(1)
    ),
    data=d
)
precis(m8h4.c)
plot(precis(m8h4.c))
```


<table class="dataframe">
<caption>A precis: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td> 1.59749659</td><td>0.5251105</td><td> 0.75826864</td><td> 2.4367245</td></tr>
	<tr><th scope=row>bA</th><td> 0.32643504</td><td>0.2521681</td><td>-0.07657837</td><td> 0.7294484</td></tr>
	<tr><th scope=row>bS</th><td> 0.12112046</td><td>0.2502008</td><td>-0.27874873</td><td> 0.5209896</td></tr>
	<tr><th scope=row>bG</th><td> 0.22185692</td><td>0.2505464</td><td>-0.17856459</td><td> 0.6222784</td></tr>
	<tr><th scope=row>bGS</th><td> 0.07076869</td><td>0.2500404</td><td>-0.32884412</td><td> 0.4703815</td></tr>
	<tr><th scope=row>sigma</th><td>11.86771030</td><td>1.0064885</td><td>10.25914729</td><td>13.4762733</td></tr>
</tbody>
</table>




    
![png](sr-ch08-output_97_1.png)
    


From this, it looks like this hypothesis is not really supported by the data.

#### 8H5
Consider the `data(wines2012)` data table. These data are expert ratings of 20 different French and American wines by 9 different American and French judges. Your goal is to model `score`, the subjective rating assigned by each judge to each wine. I recommend standardizing it. In this problem, condier only variation among judges and wines. Construct index variables of `judge` and `wine` and then use these index variables to construct a linear regression model. Justify your priors. You should the use these index variables to construct a linear regression model. Justify your priors. You should end up with 9 judge parameters and 20 wine parameters. How to you interpret the variation among the individual judges and individual wines? Do you notice any patterns, just by plotting the differences? Which judge gave the highest / lowest ratings? which wines were rated worst / best on average?



#### 8H6
Now consider three features of the winse and judges.
1. `flight`: whether the wine was white or red
1. `wine.amer`: indicator variable for American wines
1. `judge.amer`: indicator variable for American judges

Use indicator variables to model the influence of these features on the scores. Omit the individual judge and wine variables from 8H5. Do not include interaction effects yet. Again, justify your priors. What do you conclude about the differences among the wines and judges? Try to relate the results to the inferenes in the previous problem.

#### 8H7
Now consider two-way interactions among the three features. You should end up with three different interaction terms in your mode. These will be easier to buid if you use indicator variables. Agin, justify your priors. Explain what each inteaction means. By sure to interpret the model's prediction on the outcome scale (`mu`, the expected score)m not on the scale of individual parameters. You can use `link` to help with this, or just use your knowledge of the linear model instead. What do you conclude about the featurse and the scores? Can you relate the results of your models to the individual judge and wine interactions in 8H5?


