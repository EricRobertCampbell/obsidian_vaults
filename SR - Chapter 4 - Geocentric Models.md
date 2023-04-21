# Chapter 4 - Geocentric Models

Ptolemy: geocentric model. This model achieved very high levels of accuracy by employing _epicycles_ - orbits on orbits. This model was used for over 1000 years.

Of course, it is wrong. But despite that, it remains useful, as long as your goal is to establish the postition of a planet in the night sky. (In fact, this _epicycles_ idea is a form of Fourier series, so in principle, by adding more epicycles we could achieve any level of accuracy that we would like)

[[Linear Regression|*Linear Regression*]] is the geocentric model of statistics. We know it isn't right, but as long as we use it properly it can be useful.

## Why normal distributions are normal
Let's say 1000 people start on the midway line of a soccer pitch, and start to flip coins. If you get heads you move one step one way; tails, one step the other.
### Normal by addition
Let's see what happens if you do this a bunch:

```R
pos <- replicate(1000, sum(runif(16, -1, 1)))
hist(pos)
```

![png](sr-chapter4-output_2_0.png)

```R
plot(density(pos))
```

![png](sr-chapter4-output_3_0.png)

The fact that this looks normal is not surprising - if you take basically any distribution and add together a bunch of different variables from that distribution, then the distribution that you get is normal (some sort of limit theorem).

### Normal by multiplication
Here's another way to get a normal distribution. Say an organism's height is influenced by lots of alleles, and that they interact so that their effects are multiplicative rather than additive. We can simulate this as follows:


```R
library(rethinking)
growth <- replicate(1000, prod(1 + runif(12, 0, 0.1)))
dens(growth, norm.comp=T)
```

    Loading required package: rstan
    
    Loading required package: StanHeaders
    
    Loading required package: ggplot2
    
    rstan (Version 2.21.3, GitRev: 2e1f913d3ca3)
    
    For execution on a local, multicore CPU with excess RAM we recommend calling
    options(mc.cores = parallel::detectCores()).
    To avoid recompilation of unchanged Stan programs, we recommend calling
    rstan_options(auto_write = TRUE)
    
    Loading required package: cmdstanr
    
    This is cmdstanr version 0.5.0
    
    - CmdStanR documentation and vignettes: mc-stan.org/cmdstanr
    
    - Use set_cmdstan_path() to set the path to CmdStan
    
    - Use install_cmdstan() to install CmdStan
    
    Loading required package: parallel
    
    rethinking (Version 2.21)
    
    
    Attaching package: ‘rethinking’
    
    
    The following object is masked from ‘package:rstan’:
    
        stan
    
    
    The following object is masked from ‘package:stats’:
    
        rstudent
    
    



    
![png](sr-chapter4-output_5_1.png)
    


This works because for small numbers, multiplication is almost the same as addition:
$$
1.1 * 1.1 = (1 + 0.1)(1 + 0.1) = 1 + 0.2 + 0.1 * 0.1 \approx 1.2 = 1 + (0.1 + 0.1)
$$

### Normal by log-multiplication
Of course, the larger the numbers the less the product will converge to a Gaussian. Howether, their *log probabilities* do converge!

log.big <- replicate(1000, log(prod(1 + runif(12, 0, 0.5))))
dens(log.big, norm.comp=T)

This works because taking the log takes multiplication to addition!

### Using Gaussian distributions

The justification for using Gaussians so often is twofold:
1. Ontological
1. Epistemological

**Ontological**

This argument is that the world is full of Gaussian distributions. Almost everything that we measure will be influenced by lots of little bumps, and these tend to produce Gaussians.

**Epistemological**

This argument is that often, we will only know the mean and variance of a distribution for the items that we are interested in. When that is the case, then the Gaussian is the _maximum entropy distribution_. The choice of any other distribution would imply that we have additional knowledge.

One note to be careful of: the Gaussian has very thin tails - most of the density is within one sd of the mean. Many natural processes have comparatively fat tails, which means that modelling them with Gaussians will result in an underestimate of the risk of extreme events.
See: the 2008 financial crisis.

The pdf of the Gaussian is given by 

$$
p(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

This can be reparameterixed in terms of the *precision* $\tau = \frac{1}{\sigma^2}$. Simply substituting this in gives:
$$
p(x|\mu, \tau) = \frac{\tau}{\sqrt{2\pi}}e^{-\frac{\tau(x - \mu)^2}{2}}
$$

Certain Bayesian model-fitting software requires the distribution to be in this form.

## A language for describing models

Here's a general language / approach to describing statistical models:
1. Recognize a set of variables to work with Some of the variables are observable: we call these `data`. Others are unobservable things like rate or averages: we call these `parameters`.
1. We define each variable in terms of other variables,  or in terms of a probability distribution
1. The combination of variables and their probability distributions defines a _joint generative model_ that can be used both to simulate hypothetical observations as well as analyze real ones.

This approach applies to models in any field - the difficulty is in deciding which variables are important and how they should be connected - not in the mathematics itself.

Once all of these decisions have been made, we write down the model, something like this:
$$
\begin{align*}
y_1 &\sim \text{Normal($\mu_i$, $\sigma$)} \\
\mu_i &= \beta x_i \\
\beta &\sim \text{Normal(0, 10)} \\
\sigma &\sim \text{Exponential(1)} \\
x_i &\sim \text{Normal(0, 1)}\\
\end{align*}
$$

### Re-describing the globe-tossing model

Our globe-tossing model can be re-written in that form:
$$
\begin{align*}
W &\sim \text{Binomial($N$, $p$)} \\
p &\sim \text{Uniform(0, 1)}\\
\end{align*}
$$

We can read this as

> The count $W$ is distributed binomially with sample size $N$ and probability $p$. The prior for $p$ is assumed to be uniform between zero and 1.

In simple models like these, the first line defines the likelihood function for Bayes' Theorem. The other lines define priors. Both of the lines in this model are _stochastic_, as indicated by the $\sim$ symbol. A stochastic relationship is just a mapping of a variable or parameter onto a distribution. It is _stochastic_ because no single instance of the variable is known - instead, it will take on values from the distribution.

## Gaussian model of height

let's build a linear regression model. For the moment, we want a single measurement variable to model as a Gaussian distribution. There will be two parameters, the mean $\mu$ and the sd $\sigma$. Then we'll update the values of $\mu$ and $\sigma$ as the data comes in.

### The data

The data we'll work with is part of the census data for the Dobe area !Kung San, compiled by interviews by Nancy Howell in the 1960s.


```R
data(Howell1)
d <- Howell1
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>height</th><th scope=col>weight</th><th scope=col>age</th><th scope=col>male</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>151.765</td><td>47.82561</td><td>63</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>139.700</td><td>36.48581</td><td>63</td><td>0</td></tr>
	<tr><th scope=row>3</th><td>136.525</td><td>31.86484</td><td>65</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>156.845</td><td>53.04191</td><td>41</td><td>1</td></tr>
	<tr><th scope=row>5</th><td>145.415</td><td>41.27687</td><td>51</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>163.830</td><td>62.99259</td><td>35</td><td>1</td></tr>
</tbody>
</table>




```R
# We can also examine the shape 
str(d)
```

    'data.frame':	544 obs. of  4 variables:
     $ height: num  152 140 137 157 145 ...
     $ weight: num  47.8 36.5 31.9 53 41.3 ...
     $ age   : num  63 63 65 41 51 35 32 27 19 54 ...
     $ male  : int  1 0 0 1 0 1 0 1 0 1 ...



```R
# we can also use the `precis` function from `rethinking`
precis(d)
```


<table class="dataframe">
<caption>A precis: 4 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>height</th><td>138.2635963</td><td>27.6024476</td><td>81.108550</td><td>165.73500</td><td>▁▁▁▁▁▁▁▂▁▇▇▅▁</td></tr>
	<tr><th scope=row>weight</th><td> 35.6106176</td><td>14.7191782</td><td> 9.360721</td><td> 54.50289</td><td>▁▂▃▂▂▂▂▅▇▇▃▂▁</td></tr>
	<tr><th scope=row>age</th><td> 29.3443934</td><td>20.7468882</td><td> 1.000000</td><td> 66.13500</td><td>▇▅▅▃▅▂▂▁▁    </td></tr>
	<tr><th scope=row>male</th><td>  0.4724265</td><td> 0.4996986</td><td> 0.000000</td><td>  1.00000</td><td>▇▁▁▁▁▁▁▁▁▇   </td></tr>
</tbody>
</table>




```R
# We can access just the height column as a vector:
d$height
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>151.765</li><li>139.7</li><li>136.525</li><li>156.845</li><li>145.415</li><li>163.83</li><li>149.225</li><li>168.91</li><li>147.955</li><li>165.1</li><li>154.305</li><li>151.13</li><li>144.78</li><li>149.9</li><li>150.495</li><li>163.195</li><li>157.48</li><li>143.9418</li><li>121.92</li><li>105.41</li><li>86.36</li><li>161.29</li><li>156.21</li><li>129.54</li><li>109.22</li><li>146.4</li><li>148.59</li><li>147.32</li><li>137.16</li><li>125.73</li><li>114.3</li><li>147.955</li><li>161.925</li><li>146.05</li><li>146.05</li><li>152.7048</li><li>142.875</li><li>142.875</li><li>147.955</li><li>160.655</li><li>151.765</li><li>162.8648</li><li>171.45</li><li>147.32</li><li>147.955</li><li>144.78</li><li>121.92</li><li>128.905</li><li>97.79</li><li>154.305</li><li>143.51</li><li>146.7</li><li>157.48</li><li>127</li><li>110.49</li><li>97.79</li><li>165.735</li><li>152.4</li><li>141.605</li><li>158.8</li><li>155.575</li><li>164.465</li><li>151.765</li><li>161.29</li><li>154.305</li><li>145.415</li><li>145.415</li><li>152.4</li><li>163.83</li><li>144.145</li><li>129.54</li><li>129.54</li><li>153.67</li><li>142.875</li><li>146.05</li><li>167.005</li><li>158.4198</li><li>91.44</li><li>165.735</li><li>149.86</li><li>147.955</li><li>137.795</li><li>154.94</li><li>160.9598</li><li>161.925</li><li>147.955</li><li>113.665</li><li>159.385</li><li>148.59</li><li>136.525</li><li>158.115</li><li>144.78</li><li>156.845</li><li>179.07</li><li>118.745</li><li>170.18</li><li>146.05</li><li>147.32</li><li>113.03</li><li>162.56</li><li>133.985</li><li>152.4</li><li>160.02</li><li>149.86</li><li>142.875</li><li>167.005</li><li>159.385</li><li>154.94</li><li>148.59</li><li>111.125</li><li>111.76</li><li>162.56</li><li>152.4</li><li>124.46</li><li>111.76</li><li>86.36</li><li>170.18</li><li>146.05</li><li>159.385</li><li>151.13</li><li>160.655</li><li>169.545</li><li>158.75</li><li>74.295</li><li>149.86</li><li>153.035</li><li>96.52</li><li>161.925</li><li>162.56</li><li>149.225</li><li>116.84</li><li>100.076</li><li>163.195</li><li>161.925</li><li>145.415</li><li>163.195</li><li>151.13</li><li>150.495</li><li>141.605</li><li>170.815</li><li>91.44</li><li>157.48</li><li>152.4</li><li>149.225</li><li>129.54</li><li>147.32</li><li>145.415</li><li>121.92</li><li>113.665</li><li>157.48</li><li>154.305</li><li>120.65</li><li>115.6</li><li>167.005</li><li>142.875</li><li>152.4</li><li>96.52</li><li>160</li><li>159.385</li><li>149.86</li><li>160.655</li><li>160.655</li><li>149.225</li><li>125.095</li><li>140.97</li><li>154.94</li><li>141.605</li><li>160.02</li><li>150.1648</li><li>155.575</li><li>103.505</li><li>94.615</li><li>156.21</li><li>153.035</li><li>167.005</li><li>149.86</li><li>147.955</li><li>159.385</li><li>161.925</li><li>155.575</li><li>159.385</li><li>146.685</li><li>172.72</li><li>166.37</li><li>141.605</li><li>142.875</li><li>133.35</li><li>127.635</li><li>119.38</li><li>151.765</li><li>156.845</li><li>148.59</li><li>157.48</li><li>149.86</li><li>147.955</li><li>102.235</li><li>153.035</li><li>160.655</li><li>149.225</li><li>114.3</li><li>⋯</li><li>104.14</li><li>161.29</li><li>148.59</li><li>97.155</li><li>93.345</li><li>160.655</li><li>157.48</li><li>167.005</li><li>157.48</li><li>91.44</li><li>60.452</li><li>137.16</li><li>152.4</li><li>152.4</li><li>81.28</li><li>109.22</li><li>71.12</li><li>89.2048</li><li>67.31</li><li>85.09</li><li>69.85</li><li>161.925</li><li>152.4</li><li>88.9</li><li>90.17</li><li>71.755</li><li>83.82</li><li>159.385</li><li>142.24</li><li>142.24</li><li>168.91</li><li>123.19</li><li>74.93</li><li>74.295</li><li>90.805</li><li>160.02</li><li>67.945</li><li>135.89</li><li>158.115</li><li>85.09</li><li>93.345</li><li>152.4</li><li>155.575</li><li>154.305</li><li>156.845</li><li>120.015</li><li>114.3</li><li>83.82</li><li>156.21</li><li>137.16</li><li>114.3</li><li>93.98</li><li>168.275</li><li>147.955</li><li>139.7</li><li>157.48</li><li>76.2</li><li>66.04</li><li>160.7</li><li>114.3</li><li>146.05</li><li>161.29</li><li>69.85</li><li>133.985</li><li>67.945</li><li>150.495</li><li>163.195</li><li>148.59</li><li>148.59</li><li>161.925</li><li>153.67</li><li>68.58</li><li>151.13</li><li>163.83</li><li>153.035</li><li>151.765</li><li>132.08</li><li>156.21</li><li>140.335</li><li>158.75</li><li>142.875</li><li>84.455</li><li>151.9428</li><li>161.29</li><li>127.9906</li><li>160.9852</li><li>144.78</li><li>132.08</li><li>117.983</li><li>160.02</li><li>154.94</li><li>160.9852</li><li>165.989</li><li>157.988</li><li>154.94</li><li>97.9932</li><li>64.135</li><li>160.655</li><li>147.32</li><li>146.7</li><li>147.32</li><li>172.9994</li><li>158.115</li><li>147.32</li><li>124.9934</li><li>106.045</li><li>165.989</li><li>149.86</li><li>76.2</li><li>161.925</li><li>140.0048</li><li>66.675</li><li>62.865</li><li>163.83</li><li>147.955</li><li>160.02</li><li>154.94</li><li>152.4</li><li>62.23</li><li>146.05</li><li>151.9936</li><li>157.48</li><li>55.88</li><li>60.96</li><li>151.765</li><li>144.78</li><li>118.11</li><li>78.105</li><li>160.655</li><li>151.13</li><li>121.92</li><li>92.71</li><li>153.67</li><li>147.32</li><li>139.7</li><li>157.48</li><li>91.44</li><li>154.94</li><li>143.51</li><li>83.185</li><li>158.115</li><li>147.32</li><li>123.825</li><li>88.9</li><li>160.02</li><li>137.16</li><li>165.1</li><li>154.94</li><li>111.125</li><li>153.67</li><li>145.415</li><li>141.605</li><li>144.78</li><li>163.83</li><li>161.29</li><li>154.9</li><li>161.3</li><li>170.18</li><li>149.86</li><li>123.825</li><li>85.09</li><li>160.655</li><li>154.94</li><li>106.045</li><li>126.365</li><li>166.37</li><li>148.2852</li><li>124.46</li><li>89.535</li><li>101.6</li><li>151.765</li><li>148.59</li><li>153.67</li><li>53.975</li><li>146.685</li><li>56.515</li><li>100.965</li><li>121.92</li><li>81.5848</li><li>154.94</li><li>156.21</li><li>132.715</li><li>125.095</li><li>101.6</li><li>160.655</li><li>146.05</li><li>132.715</li><li>87.63</li><li>156.21</li><li>152.4</li><li>162.56</li><li>114.935</li><li>67.945</li><li>142.875</li><li>76.835</li><li>145.415</li><li>162.56</li><li>156.21</li><li>71.12</li><li>158.75</li></ol>




```R
# We'll just be working with adults (since there is a strong correlation between age and height 
# which will swamp out the relationship we are looking for between between the height and weight)
d2 <- d[d$age >= 18,]
head(d2)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>height</th><th scope=col>weight</th><th scope=col>age</th><th scope=col>male</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>151.765</td><td>47.82561</td><td>63</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>139.700</td><td>36.48581</td><td>63</td><td>0</td></tr>
	<tr><th scope=row>3</th><td>136.525</td><td>31.86484</td><td>65</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>156.845</td><td>53.04191</td><td>41</td><td>1</td></tr>
	<tr><th scope=row>5</th><td>145.415</td><td>41.27687</td><td>51</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>163.830</td><td>62.99259</td><td>35</td><td>1</td></tr>
</tbody>
</table>



recall that to access the elements of a dataframe, you use `d[row, col]`. 

### The model

Let's look at the density:


```R
dens(d2$height)
```


    
![png](sr-chapter4-output_15_0.png)
    


So it looks like this could be apporimately normally distributed, so it seems reasonable to model it that way.

$$
h_i \sim \text{Normal}(\mu,\ \sigma)
$$

Of course, we also need to do something with the parameters $\mu$ and $\sigma$. Let's assign some priors:
$$
\begin{align*}
h_i &\sim \text{Normal}(\mu,\ \sigma) \\
\mu &\sim \text{Normal}(178,\ 20) \\
\sigma &\sim \text{Uniform}(0,\ 50) \\
\end{align*}
$$

Where did these values come from? Well, the author of the textbook is 178cm tall, and the range 138 -- 218 ($\pm 2\sigma$) encompasses the plausible range of human heights. Here, we see that we actually need some domain knowledge to come up with the prior.

Let's plot the priors to get an idea of what the model is starting out with:


```R
curve(dnorm(x, 178, 20), from=100, to=250)
```


    
![png](sr-chapter4-output_17_0.png)
    



```R
curve(dunif(x, 0, 50), from=-10, to=60)
```


    
![png](sr-chapter4-output_18_0.png)
    


Now let's do a _prior predictive check_ --- that is, let's see what predictions our priors make:


```R
sample_mu <- rnorm(1e4, 178, 20)
sample_sigma <- runif(1e4, 0, 50)
prior_h <- rnorm(1e4, sample_mu, sample_sigma)
dens(prior_h)
```


    
![png](sr-chapter4-output_20_0.png)
    


Looking at this, we can see that our priors include predictions with quite fat tails: we would be very surprised to see someone 300cm tall, but here there is a surprisingly high likelihood. We should be careful about making our priors too uninformative --- sometimes it can lead to physically impossible results. For instance, let's see what happens if we use a less informative prior for the heights.


```R
sample_mu <- rnorm(1e4, 178, 100)
sample_sigma <- runif(1e4, 0, 50)
prior_h <- rnorm(1e4, sample_mu, sample_sigma)
dens(prior_h)
```


    
![png](sr-chapter4-output_22_0.png)
    

Here we see that lots of people are expected to have negative height, and others are much taller than the height of the tallest person ever recorded (272cm). Here we have lots of data, so having silly priors isn't that damaging, but that won't always be the case.

### Grid approximation of the posterior distribution

Here's a way to do this with a grid approximation!

```R
mu.list <- seq(from=150, to=160, length.out=100)
sigma.list <- seq(from=7, to=9, length.out=100)
post <- expand.grid(mu=mu.list, sigma=sigma.list)
post$LL <- sapply(1:nrow(post), function(i)
    sum(
        dnorm(d2$height, post$mu[i], post$sigma[i], log=TRUE)
    )
)
post$prod <- post$LL + dnorm(post$mu, 178, 20, TRUE) + dunif(post$sigma, 0, 50, TRUE)
post$prob <- exp(post$prod - max(post$prod))
```


```R
contour_xyz(post$mu, post$sigma, post$prob)
```


    
![png](sr-chapter4-output_26_0.png)
    



```R
image_xyz(post$mu, post$sigma, post$prob)
```


    
![png](sr-chapter4-output_27_0.png)
    


### Sampling from the posterior

Of course, we can also use the very general method of sampling from the distribution. The trick is that here we are sampling from the rows in proportion to the likelihood of that row occurring:


```R
sample.rows <- sample(1:nrow(post), size=1e4, replace=TRUE, prob=post$prob)
sample.mu <- post$mu[sample.rows]
sample.sigma <- post$sigma[sample.rows]
```


```R
plot(sample.mu, sample.sigma, cex=0.5, pch=16, col=col.alpha(rangi2,0.1))
```


    
![png](sr-chapter4-output_30_0.png)
    



```R
# Now the marginal distributions for mu and sigma
dens(sample.mu)
dens(sample.sigma)
```


    
![png](sr-chapter4-output_31_0.png)
    



    
![png](sr-chapter4-output_31_1.png)
    



```R
# Posterior compatibility intervals
PI(sample.mu)
PI(sample.sigma)
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>5%</dt><dd>153.939393939394</dd><dt>94%</dt><dd>155.252525252525</dd></dl>




<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>5%</dt><dd>7.32323232323232</dd><dt>94%</dt><dd>8.23232323232323</dd></dl>



### Finding the posterior distribution with `quap`

Now we'll abandon the grid approximation and use a *quadratic approximation* instead.

The `quap` function from the `rethinking` library uses the model definition we used earlier.


```R
d <- Howell1
d2 <- d[d$age >= 18,]

# now we write the model as an alist
flist <- alist(
    height ~ dnorm(mu, sigma),
    mu ~ dnorm(178, 20),
    sigma ~ dunif(0, 50)
)

# fit the model to the data
m4.1 <- quap(flist, data=d2)
```


```R
precis(m4.1)
```


<table class="dataframe">
<caption>A precis: 2 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>mu</th><td>154.607023</td><td>0.4119947</td><td>153.948576</td><td>155.265471</td></tr>
	<tr><th scope=row>sigma</th><td>  7.731333</td><td>0.2913860</td><td>  7.265642</td><td>  8.197024</td></tr>
</tbody>
</table>



The priors that we used were very weak; let's see what happens when we use a more informative prior for $\mu$:


```R
m4.2 <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu ~ dnorm(178, 0.1),
        sigma ~ dunif(0, 50)
    ),
    data=d2,
)
precis(m4.2)
```


<table class="dataframe">
<caption>A precis: 2 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>mu</th><td>177.86375</td><td>0.1002354</td><td>177.70356</td><td>178.02395</td></tr>
	<tr><th scope=row>sigma</th><td> 24.51756</td><td>0.9289235</td><td> 23.03297</td><td> 26.00216</td></tr>
</tbody>
</table>



We used a very strong prior, so $mu$ has barely changed from the prior. However, $\sigma$ has changed! This is because the certainty about $mu$ changes the values for $\sigma$.

### Sampling from `quap`
The above explains how to use `quap` to get the quadratic approximation, but we'd also like to sample from it. How do we do this?


```R
# Get the variance / covariance matrix for the resulting distribution:
vcov(m4.1)
```


<table class="dataframe">
<caption>A matrix: 2 × 2 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>mu</th><th scope=col>sigma</th></tr>
</thead>
<tbody>
	<tr><th scope=row>mu</th><td>0.1697395929</td><td>0.0002180298</td></tr>
	<tr><th scope=row>sigma</th><td>0.0002180298</td><td>0.0849058006</td></tr>
</tbody>
</table>



We can decompose it into two elements:
1. A vector of variances for the parameters, and
2. A correlation matrix that tells us how changes in any parameter lead to correlated changes in the others.


```R
diag(vcov(m4.1))
cov2cor(vcov(m4.1))
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>mu</dt><dd>0.169739592874664</dd><dt>sigma</dt><dd>0.0849058006414092</dd></dl>




<table class="dataframe">
<caption>A matrix: 2 × 2 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>mu</th><th scope=col>sigma</th></tr>
</thead>
<tbody>
	<tr><th scope=row>mu</th><td>1.000000000</td><td>0.001816167</td></tr>
	<tr><th scope=row>sigma</th><td>0.001816167</td><td>1.000000000</td></tr>
</tbody>
</table>



Notice that the covariances are near 0 - this tells us that for this simple model, $\sigma$ and $\mu$ are close to independent.


```R
# use the extract.samples function from rethinking to grab values:
post <- extract.samples(m4.1, n=1e4)
head(post)
```


<table class="dataframe">
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>mu</th><th scope=col>sigma</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>154.8004</td><td>7.426072</td></tr>
	<tr><th scope=row>2</th><td>154.0160</td><td>8.084437</td></tr>
	<tr><th scope=row>3</th><td>154.5143</td><td>7.600915</td></tr>
	<tr><th scope=row>4</th><td>154.7694</td><td>7.485661</td></tr>
	<tr><th scope=row>5</th><td>154.4182</td><td>7.686298</td></tr>
	<tr><th scope=row>6</th><td>154.1857</td><td>7.604763</td></tr>
</tbody>
</table>




```R
# These values should be close to the values from before:
precis(post)
```


<table class="dataframe">
<caption>A precis: 2 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>mu</th><td>154.610070</td><td>0.4130550</td><td>153.957270</td><td>155.26536</td><td>▁▁▁▅▇▂▁▁   </td></tr>
	<tr><th scope=row>sigma</th><td>  7.732029</td><td>0.2946993</td><td>  7.257321</td><td>  8.20631</td><td>▁▁▁▂▅▇▇▃▁▁▁</td></tr>
</tbody>
</table>



## Linear prediction

We now have a model of the heights in a population of adults. Typically, we are interested in modeling how an outcome is related to some other variable, a *predictor variable*. If the predictor variable has any statistical association with the outcome variable, then we can use it to predict the outcome.

So now let's look at how the height covaries with weight.


```R
data(Howell1); d <- Howell1; d2 <- d[d$age >= 18,]
plot(d2$height ~ d2$weight)
```


    
![png](sr-chapter4-output_46_0.png)
    


Looking at this, it seems clear that there is some sort of relationship! But how do we take the next step to incorporate what we know about about model so that it can make predictions?

### The linear model strategy

The strategy is to make the parameter for the mean of a Gaussian distribution, $\mu$, into a linear function of the predictor variable and other, new parameters that we invent. This strategy is often called the *linear model*.

Let $x$ be the name for the column of weight measurements. Then our new model for predicting height as a function of weight is:

$$
\begin{align*}
h_i &\sim \text{Normal}(\mu_i, \sigma) & \text{[likelihood]} \\
\mu_i &= \alpha + \beta(x_i - \bar{x}) & \text{[linear model]} \\
\alpha &\sim \text{Normal}(178, 20) \\
\beta &\sim \text{Normal(0, 10)} \\
\sigma &\sim \text{Uniform}(0, 50) \\
\end{align*}
$$

#### Probability of the data

Notice that now, each height depends on its own particular $\mu_i$, which in turn depends on the particular $x_i$.

#### Linear model

This means that now $\mu_i$ is no longer a parameter to be estimated, but is now constructed from the other parameters, particularly $\alpha$ and $\beta$. Notice that this is not a stochastic relationship - $=$ rather than $\sim$. The parameters $\alpha$ and $\beta$ are ones that we made up to control how the mean $\mu_i$ can vary across the data.

In order to understand our priors, let's simulate values from them:


```R
set.seed(2971)

# simulating 100 lines drawn from the prior
N <- 100
a <- rnorm(N, 178, 20)
b <- rnorm(N, 0, 10)

# Now plot all 100 lines
plot(NULL, xlim=range(d2$weight), ylim=c(-100, 400),
     xlab="eright", ylab="height",
    )
abline(h=0, lty=2)
abline(h=272, lty=1, lwd=0.5)
mtext("b ~ dnorm(0,10)")
xbar <- mean(d2$weight)
for ( i in 1:N ) {
    curve(a[i] + b[i] * (x - xbar),
          from=min(d2$weight),
          to=max(d2$weight),
          add=T,
          col=col.alpha("black", 0.2),
      )
}
```


    
![png](sr-chapter4-output_48_0.png)
    


Notice that this is not really believable - the line at 272 is the world's tallest person, and anything below zero seems$\dots$ unlikely. We can definitely do better!

First thing: we expect that the relationship between weight and height to be positive. One way to do that is to ensure that the slope parameter $\beta$ is positive by drawing it from a Log-Normal distribution. Basically, this would mean that the log of that values of $\beta$ would be normally distributed. For our model:
$$
\beta \sim \text{Log-Normal}(0,1)
$$


```R
b <- rlnorm(1e4,0,1)
dens(b, xlim=c(0,5), adj=0.1)
```


    
![png](sr-chapter4-output_50_0.png)
    



```R
# now let's see what happens to our prior predictions
set.seed(2971)

# simulating 100 lines drawn from the prior
N <- 100
a <- rnorm(N, 178, 20)
# we changed this to a log-normal prior and changed the sigma from 10 to 1
b <- rlnorm(N, 0, 1)

# Now plot all 100 lines
plot(NULL, xlim=range(d2$weight), ylim=c(-100, 400),
     xlab="weight", ylab="height",
    )
abline(h=0, lty=2)
abline(h=272, lty=1, lwd=0.5)
mtext("b ~ dnorm(0,10)")
xbar <- mean(d2$weight)
for ( i in 1:N ) {
    curve(a[i] + b[i] * (x - xbar),
          from=min(d2$weight),
          to=max(d2$weight),
          add=T,
          col=col.alpha("black", 0.2),
      )
}
```


    
![png](sr-chapter4-output_51_0.png)
    


Right away we can see that this is far more credible.

### Finding the posterior distribution

The code to find the posterior is very similar to what we already did.


```R
xbar <- mean(d2$weight)

# fit the model
m4.3 <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu <- a + b * (weight - xbar),
        a ~ dnorm(178, 20),
        b ~ dlnorm(0, 1),
        sigma ~ dunif(0, 50)
    ),
    data=d2,
)
```

### Interpreting the posterior distribution

One trouble with statistical models is that they can be hard to interpret.

There are two broad categories of processing:
1. Reading tables
1. Plotting simulations

For simple models, sometimes you can glean a lot of information from a table of numbers. But as models become more complex, simulations and graphs and whatnot are going to be the way to go. Plotting the predictions of your models can also allow you to inquire about things that are hard to read from tables:
1. Whether or not the model fitting process worked correctly
1. The *absolute* magnitude, rather than the merely *relative* magnitude, of a relationship between outcome and predictor
1. The uncertainty surrounding an average relationship
1. The uncertainty surrounding the implied predictions of the model, as these are distinct from mere parameter uncertainty


```R
precis(m4.3)
```


<table class="dataframe">
<caption>A precis: 3 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>154.6013671</td><td>0.27030766</td><td>154.1693633</td><td>155.0333710</td></tr>
	<tr><th scope=row>b</th><td>  0.9032807</td><td>0.04192363</td><td>  0.8362787</td><td>  0.9702828</td></tr>
	<tr><th scope=row>sigma</th><td>  5.0718809</td><td>0.19115478</td><td>  4.7663786</td><td>  5.3773831</td></tr>
</tbody>
</table>



If we look at the value for $b$ ($\beta$), we see that plausible values are around 0.9. We should think of this as saying that according to our model, someone 1kg heavier is expected to be about 0.9cm taller.


```R
round(vcov(m4.3), 3)
```


<table class="dataframe">
<caption>A matrix: 3 × 3 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>a</th><th scope=col>b</th><th scope=col>sigma</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>0.073</td><td>0.000</td><td>0.000</td></tr>
	<tr><th scope=row>b</th><td>0.000</td><td>0.002</td><td>0.000</td></tr>
	<tr><th scope=row>sigma</th><td>0.000</td><td>0.000</td><td>0.037</td></tr>
</tbody>
</table>



So we actually see almost no covariance among the different variables.


```R
pairs(m4.3)
```


    
![png](sr-chapter4-output_59_0.png)
    


#### Plotting the posterior inference against the data

It's almost always better to plot the posterior inference against the data. This both helps with interpreting the posterior and can serve as an informal check of model validity.

Let's start with a simple one: the data, and one line


```R
plot(height ~ weight, data=d2, col=rangi2)
post <- extract.samples(m4.3)
a_map <- mean(post$a)
b_map <- mean(post$b)
curve(a_map + b_map * (x - xbar), add=T)
```


    
![png](sr-chapter4-output_61_0.png)
    


Each point is an individual, and the line here is formed from the mean of $a$ and $b$.

#### Adding uncertainty around the mean

The mean line (the one that we plotted) is really just the single most plausible line. There are lots of other ones as well!

To do this, let's take a look at the samples:


```R
post <- extract.samples(m4.3)
post[1:5,]
```


<table class="dataframe">
<caption>A data.frame: 5 × 3</caption>
<thead>
	<tr><th></th><th scope=col>a</th><th scope=col>b</th><th scope=col>sigma</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>154.5789</td><td>0.9376825</td><td>5.220756</td></tr>
	<tr><th scope=row>2</th><td>154.4067</td><td>0.8937310</td><td>4.752735</td></tr>
	<tr><th scope=row>3</th><td>154.4622</td><td>0.9150822</td><td>5.341227</td></tr>
	<tr><th scope=row>4</th><td>154.2649</td><td>0.9236067</td><td>5.160423</td></tr>
	<tr><th scope=row>5</th><td>155.1258</td><td>0.9495934</td><td>5.108891</td></tr>
</tbody>
</table>



Let's add the uncertainty back in. We'll do this without all of the data, so that we can see the effect of adding more data in has on the level of certainty


```R
for (N in c(10, 50, 150, 352)) {
    dN <- d2[1:N,]
    mN <- quap(
        alist(
            height ~ dnorm(mu, sigma),
            mu <- a + b * (weight - mean(weight)),
            a ~ dnorm(178, 20),
            b ~ dlnorm(0, 1),
            sigma ~ dunif(0, 50)
        ),
        data=dN
    )

    # Now choose 20 lines and plot them
    post <- extract.samples(mN, n=20)

    plot(dN$weight, dN$height, xlim=range(d2$weight), ylim=range(d2$height), col=rangi2, xlab="weight", ylab="height")
    mtext(concat("N = ", N))

    # plot the lines, with transparency
    for (i in 1:20) {
        curve(
            post$a[i] + post$b[i] * (x - mean(dN$weight)),
            col=col.alpha("black", 0.3), add=TRUE
        )
    }
}

```


    
![png](sr-chapter4-output_65_0.png)
    



    
![png](sr-chapter4-output_65_1.png)
    



    
![png](sr-chapter4-output_65_2.png)
    



    
![png](sr-chapter4-output_65_3.png)
    


So we can see that as we get more information, we become more confident about our predictions.

#### Plotting regression intervals and contours

Often, a more informative approach to what we just did is to plot some sort of interval around the mean line.

Focus for a moment on a single individual, weighing (say) 50kg. We can quickly make a list of 10 000 values of $\mu$ for this individual:


```R
post <- extract.samples(m4.3)
mu_at_50 <- post$a + post$b * (50 - xbar)
mu_at_50
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>159.345662478099</li><li>159.300840212747</li><li>158.605871494369</li><li>159.145049431996</li><li>159.287968519242</li><li>158.827819877345</li><li>158.602388144236</li><li>159.08674561206</li><li>159.720802634477</li><li>159.382819735151</li><li>159.053774161622</li><li>159.277506907018</li><li>159.11425261819</li><li>158.334763639823</li><li>159.099312586467</li><li>158.982900978314</li><li>159.147846295161</li><li>158.991717347616</li><li>159.198508549667</li><li>158.845742236956</li><li>159.82195201956</li><li>159.181023414794</li><li>159.53417683862</li><li>158.6001369098</li><li>158.410594506441</li><li>158.746806128515</li><li>159.85370953244</li><li>159.337411486328</li><li>159.020752602127</li><li>158.503265700495</li><li>159.504879989751</li><li>159.1258286159</li><li>159.048422113101</li><li>159.036574387715</li><li>158.785221977478</li><li>158.898599854221</li><li>159.645759444335</li><li>159.420809490049</li><li>159.86323338237</li><li>158.773651138299</li><li>159.317649523789</li><li>159.100257972895</li><li>158.837101588998</li><li>159.284499839601</li><li>159.69930317058</li><li>159.262999278317</li><li>159.548434763414</li><li>159.230930385224</li><li>158.725697651865</li><li>159.243895921448</li><li>159.438343229396</li><li>159.169512481554</li><li>159.512682220713</li><li>158.970979161417</li><li>158.78929031869</li><li>159.232520798275</li><li>159.345765097482</li><li>159.328703410333</li><li>158.698823309493</li><li>158.085200682379</li><li>159.052187193651</li><li>159.44574858549</li><li>158.66584856812</li><li>159.056076946884</li><li>159.72311913394</li><li>158.776775903294</li><li>159.330901659879</li><li>159.277323697498</li><li>159.201082682365</li><li>159.103759832857</li><li>159.057463448672</li><li>159.071256141959</li><li>159.325271143241</li><li>159.540105530895</li><li>159.015867939954</li><li>159.322658253304</li><li>158.656307772418</li><li>159.294469635702</li><li>158.884341286281</li><li>159.378751403519</li><li>159.352020093052</li><li>159.644028884328</li><li>158.429286850754</li><li>159.213147143523</li><li>158.705786347072</li><li>158.612371383775</li><li>159.304321682279</li><li>158.929929097482</li><li>158.324806547296</li><li>158.395798543728</li><li>159.203739732836</li><li>158.826970920387</li><li>159.427900711451</li><li>158.9989226425</li><li>159.066687233231</li><li>158.916628039459</li><li>159.852380682977</li><li>158.912544594522</li><li>158.96536198854</li><li>158.77832654252</li><li>158.998989290075</li><li>158.993100309278</li><li>159.364889126436</li><li>158.680167638727</li><li>159.011290698473</li><li>159.476776286307</li><li>159.170897204395</li><li>159.04964511669</li><li>158.803364618754</li><li>158.349482124356</li><li>158.90351293174</li><li>159.344673910763</li><li>158.621216034622</li><li>159.165763901321</li><li>159.452875699017</li><li>159.450054085702</li><li>159.2290596073</li><li>158.839088126623</li><li>158.776802034021</li><li>158.598613967204</li><li>158.867331850817</li><li>159.249575766064</li><li>159.047871636413</li><li>158.455454334549</li><li>159.129299480677</li><li>159.352397519689</li><li>158.916267971215</li><li>159.129787001979</li><li>159.353092566187</li><li>159.853551172036</li><li>159.345197309429</li><li>158.807485507514</li><li>159.002381742168</li><li>159.419290654346</li><li>159.169873540079</li><li>159.27957523098</li><li>159.379643423908</li><li>159.017634555268</li><li>159.501096473278</li><li>159.272228631902</li><li>159.137709878049</li><li>159.499365785969</li><li>159.032307034081</li><li>159.217776627623</li><li>159.065421951095</li><li>159.275952122516</li><li>159.620159092867</li><li>158.703785572468</li><li>159.387470478084</li><li>159.331739332385</li><li>158.824733745012</li><li>158.784692452127</li><li>159.16862541241</li><li>159.209699920365</li><li>158.562271900631</li><li>158.576186984776</li><li>159.155120245373</li><li>159.13250792041</li><li>159.21226259172</li><li>158.913189279411</li><li>159.320503992374</li><li>159.575866526827</li><li>159.491748168137</li><li>159.006785588869</li><li>159.065156637378</li><li>158.562964400728</li><li>158.955225361579</li><li>158.851363819656</li><li>158.795604132654</li><li>159.184046530252</li><li>159.315293103929</li><li>159.13228244484</li><li>158.984759701984</li><li>159.592830085592</li><li>158.933004783482</li><li>158.824595951443</li><li>159.128641150507</li><li>159.08821455636</li><li>159.254767015699</li><li>158.57358020348</li><li>159.100338703037</li><li>159.356808495891</li><li>158.973061343733</li><li>159.246825397947</li><li>158.162582928985</li><li>159.326192155822</li><li>159.47515657118</li><li>159.01682320119</li><li>158.602216430261</li><li>159.448896173848</li><li>159.044076830521</li><li>159.832162414452</li><li>158.888330258339</li><li>159.357585281241</li><li>158.977417712539</li><li>159.854404090115</li><li>159.308711917689</li><li>158.87069483186</li><li>158.930828582608</li><li>159.046468707911</li><li>⋯</li><li>159.186571972103</li><li>158.876888010477</li><li>159.332307662876</li><li>158.639811075863</li><li>159.017482208367</li><li>159.391261467511</li><li>158.694367296936</li><li>159.273973271492</li><li>159.144935297962</li><li>158.973973500631</li><li>158.965997817129</li><li>159.098128944782</li><li>159.112056084686</li><li>159.252527020277</li><li>159.2632303027</li><li>159.43499600348</li><li>159.20720844194</li><li>159.104644103474</li><li>159.451095866851</li><li>159.277553773675</li><li>158.797128277818</li><li>159.24444678107</li><li>159.232366918136</li><li>159.067645241576</li><li>159.713971089061</li><li>159.313216372407</li><li>158.578982846059</li><li>159.278747335233</li><li>159.288299947132</li><li>159.167481557468</li><li>159.177669466306</li><li>158.678883266174</li><li>159.214035332332</li><li>159.444155594589</li><li>159.012115432728</li><li>158.628779970602</li><li>159.430723575879</li><li>158.712619161099</li><li>159.076888793211</li><li>158.759396542967</li><li>159.226237640612</li><li>158.79906458668</li><li>159.227356510668</li><li>159.065066505582</li><li>159.136620206597</li><li>158.732869589011</li><li>159.202154178316</li><li>158.774458467438</li><li>159.075009679864</li><li>159.761111177525</li><li>159.342806122452</li><li>158.855819697962</li><li>159.229539161102</li><li>158.763229835086</li><li>159.103206626968</li><li>159.294343156032</li><li>159.003616538744</li><li>158.584341122919</li><li>159.391494739036</li><li>159.518534701087</li><li>158.909936762991</li><li>159.118167721313</li><li>158.691984918439</li><li>159.09222454912</li><li>159.509244832273</li><li>159.257680896489</li><li>159.548528498442</li><li>159.38975112826</li><li>158.964452126375</li><li>159.048017353233</li><li>158.943884032974</li><li>159.126461783144</li><li>159.224186307878</li><li>159.165450487262</li><li>158.812311023868</li><li>159.204240525466</li><li>158.475568781299</li><li>158.961137799311</li><li>158.822675090434</li><li>159.234624120918</li><li>158.786097815157</li><li>159.217434679678</li><li>159.279799090147</li><li>158.901754816408</li><li>158.859760503775</li><li>158.834586099983</li><li>159.228855392341</li><li>158.532373209429</li><li>158.727545963637</li><li>158.570693285728</li><li>158.97672240726</li><li>159.064760551975</li><li>159.41955784304</li><li>158.96744485261</li><li>158.839754016053</li><li>158.984543746791</li><li>159.105507806368</li><li>159.242121594674</li><li>159.191622568501</li><li>159.091063817185</li><li>159.03255203331</li><li>159.027581651901</li><li>159.041670723154</li><li>159.040004619565</li><li>159.022502519419</li><li>159.119307687844</li><li>158.9907839344</li><li>159.020432422546</li><li>159.39882175362</li><li>158.992074876784</li><li>159.352603675948</li><li>158.400246609889</li><li>159.11686314706</li><li>158.643175314302</li><li>158.969743289462</li><li>158.961956478789</li><li>159.398904634098</li><li>159.06943084266</li><li>159.252872299643</li><li>158.843645619181</li><li>159.006665327141</li><li>159.664765164389</li><li>158.802013948954</li><li>158.869659451897</li><li>158.789296472586</li><li>159.605245274779</li><li>158.431099778129</li><li>159.188481755094</li><li>158.949129486978</li><li>159.563571013149</li><li>158.929939874363</li><li>158.74197901771</li><li>158.848592825004</li><li>158.292825534026</li><li>158.624009773081</li><li>158.959317102676</li><li>159.111396782904</li><li>159.42212361109</li><li>158.798306823468</li><li>158.707189189636</li><li>158.845305842325</li><li>159.396410013783</li><li>158.749949490308</li><li>159.285852146028</li><li>159.452236013531</li><li>159.163270628406</li><li>158.881629700234</li><li>159.206177093987</li><li>158.859411913398</li><li>159.009958887767</li><li>158.513598038661</li><li>158.485467929875</li><li>159.449664294235</li><li>159.464215578532</li><li>158.604465574412</li><li>159.673977674921</li><li>158.422723343579</li><li>158.435887241556</li><li>159.49536175372</li><li>159.243801165208</li><li>159.040450607627</li><li>159.215733017383</li><li>159.210472350654</li><li>159.079046107746</li><li>159.827503776503</li><li>159.285418662939</li><li>159.12581714562</li><li>159.928352356696</li><li>159.100441048096</li><li>160.025524674421</li><li>159.191659414756</li><li>158.626218226169</li><li>159.07963922882</li><li>159.322286823972</li><li>159.570005841174</li><li>158.876460840443</li><li>159.614138909926</li><li>159.380752246192</li><li>159.208146313627</li><li>159.155009483866</li><li>158.713729575731</li><li>159.531534177186</li><li>159.017114464203</li><li>160.068379302488</li><li>159.050399226462</li><li>159.275749478508</li><li>159.061206263379</li><li>158.153249737912</li><li>159.215294425138</li><li>159.171548450794</li><li>159.023809182298</li><li>159.191950426374</li><li>159.631093336847</li><li>158.946768229786</li><li>159.641746294327</li><li>159.270148069634</li><li>159.332812824507</li><li>159.244411389895</li><li>158.894792100656</li><li>158.988330117315</li></ol>




```R
dens(mu_at_50, col=rangi2, lwd=2, xlab="mu|weight=50")
```

![png](sr-chapter4-output_68_0.png)

Since this is a distribution, we can find intervals for it, same as for any distribution:

```R
PI(mu_at_50, prob=0.89)
```

<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>5%</dt><dd>158.568328109282</dd><dt>94%</dt><dd>159.664718153956</dd></dl>



To do this for all values (not just 50), we can use the `link` function from `rethinking`. What it does is to take the `quap` approximation, sample from the posterior distribution, and then compute $\mu$ for each case in the data and sample from the posterior distribution.


```R
mu <- link(m4.3)
str(mu)
```

     num [1:1000, 1:352] 157 157 157 157 158 ...


So what we end up with is a big matrix of values for $\mu$. Each column is a case (row) in the data. Since there are 352 rows in $d2$, there are 352 columns in the matric $mu$ above.


```R
# define sequences of weights to compute predictions for
# these values will be on the horizontal axis
weight.seq <- seq(from=25, to=70, by=1)

# use link to compute mu
# for each sample from posterior
# and for each weight in weight.seq
mu <- link( m4.3, data=data.frame(weight=weight.seq))
str(mu)
```

     num [1:1000, 1:46] 138 136 137 136 137 ...


Now there are only 46 columns, since we only fed it 46 input values.

The final step is to summarize the distribution for each weight value. We'll up `apply`, which applies a function of your choice to a matrix


```R
# summarize the distributions of mu
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob=0.89)
```

```R
aply(mu, 2, mean)
```
Should be read as 'Apply the function `mean` to each column (dimension "2") of the matrix'


```R
mu.mean
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>136.522299416045</li><li>137.427010581551</li><li>138.331721747056</li><li>139.236432912562</li><li>140.141144078068</li><li>141.045855243574</li><li>141.950566409079</li><li>142.855277574585</li><li>143.759988740091</li><li>144.664699905597</li><li>145.569411071102</li><li>146.474122236608</li><li>147.378833402114</li><li>148.28354456762</li><li>149.188255733125</li><li>150.092966898631</li><li>150.997678064137</li><li>151.902389229643</li><li>152.807100395148</li><li>153.711811560654</li><li>154.61652272616</li><li>155.521233891666</li><li>156.425945057171</li><li>157.330656222677</li><li>158.235367388183</li><li>159.140078553689</li><li>160.044789719194</li><li>160.9495008847</li><li>161.854212050206</li><li>162.758923215711</li><li>163.663634381217</li><li>164.568345546723</li><li>165.473056712229</li><li>166.377767877734</li><li>167.28247904324</li><li>168.187190208746</li><li>169.091901374252</li><li>169.996612539757</li><li>170.901323705263</li><li>171.806034870769</li><li>172.710746036275</li><li>173.61545720178</li><li>174.520168367286</li><li>175.424879532792</li><li>176.329590698298</li><li>177.234301863803</li></ol>




```R
mu.PI
```


<table class="dataframe">
<caption>A matrix: 2 × 46 of type dbl</caption>
<tbody>
	<tr><th scope=row>5%</th><td>135.0892</td><td>136.0616</td><td>137.0262</td><td>137.9920</td><td>138.9579</td><td>139.9237</td><td>140.8939</td><td>141.8541</td><td>142.8196</td><td>143.7910</td><td>⋯</td><td>167.9757</td><td>168.8244</td><td>169.6622</td><td>170.4931</td><td>171.3282</td><td>172.1585</td><td>172.9925</td><td>173.8323</td><td>174.6725</td><td>175.5184</td></tr>
	<tr><th scope=row>94%</th><td>137.8951</td><td>138.7263</td><td>139.5753</td><td>140.4229</td><td>141.2775</td><td>142.1341</td><td>142.9903</td><td>143.8280</td><td>144.6656</td><td>145.5149</td><td>⋯</td><td>170.2506</td><td>171.2268</td><td>172.1978</td><td>173.1622</td><td>174.1323</td><td>175.1177</td><td>176.0986</td><td>177.0539</td><td>178.0191</td><td>178.9851</td></tr>
</tbody>
</table>




```R
# plot raw data, fading out the points to make line and interval more visible
plot(height ~ weight, data=d2, col=col.alpha(rangi2,0.5))

# plot the MAP lines, aka the mean mu for each weight
lines(weight.seq, mu.mean)

# plot a shaded region for 89% PI
shade(mu.PI, weight.seq)
```

![png](sr-chapter4-output_80_0.png)

To summarize, here's the recipe for generating predictions and intervals from the posterior of a fit model:
1. Use `link` to generate distributions of posterior values of $\mu$. The default behaviour of `link` is to use the original data, so you have to pass it a list of new horizontal ais values you want to plot posterior predictions across
1. Use summary functions like `mean` or `PI` to find averages and lower and upper bounds of $\mu$ for each value of the predictor variable
1. Finally, use plotting functions like `lines` and `shade` to draw the lines and intervals. Or, you might plot the distributions of the predictions, or do further numerical calculations with them

** How `link` works **

The `link` function is not terribly complicated. Here's how it might look in our case:

```R
post <- extract.samples(m4.3)
mu.link <- function(weight) post$a + post$b * (weight - xbar)
weight.seq <- seq(from=25, to=70, by=1)
mu <- sapply(weight.seq, mu.link)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI)
```


```R
mu.mean
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>136.551938147591</li><li>137.455022774414</li><li>138.358107401237</li><li>139.26119202806</li><li>140.164276654883</li><li>141.067361281706</li><li>141.970445908528</li><li>142.873530535351</li><li>143.776615162174</li><li>144.679699788997</li><li>145.58278441582</li><li>146.485869042643</li><li>147.388953669466</li><li>148.292038296288</li><li>149.195122923111</li><li>150.098207549934</li><li>151.001292176757</li><li>151.90437680358</li><li>152.807461430403</li><li>153.710546057226</li><li>154.613630684048</li><li>155.516715310871</li><li>156.419799937694</li><li>157.322884564517</li><li>158.22596919134</li><li>159.129053818163</li><li>160.032138444986</li><li>160.935223071809</li><li>161.838307698631</li><li>162.741392325454</li><li>163.644476952277</li><li>164.5475615791</li><li>165.450646205923</li><li>166.353730832746</li><li>167.256815459569</li><li>168.159900086391</li><li>169.062984713214</li><li>169.966069340037</li><li>170.86915396686</li><li>171.772238593683</li><li>172.675323220506</li><li>173.578407847329</li><li>174.481492474151</li><li>175.384577100974</li><li>176.287661727797</li><li>177.19074635462</li></ol>




```R
mu.PI
```


<table class="dataframe">
<caption>A matrix: 2 × 46 of type dbl</caption>
<tbody>
	<tr><th scope=row>5%</th><td>135.1395</td><td>136.1057</td><td>137.0768</td><td>138.0440</td><td>139.0118</td><td>139.9768</td><td>140.9429</td><td>141.9087</td><td>142.8728</td><td>143.8376</td><td>⋯</td><td>167.9067</td><td>168.7443</td><td>169.5822</td><td>170.4206</td><td>171.2634</td><td>172.1023</td><td>172.9452</td><td>173.7862</td><td>174.6224</td><td>175.4604</td></tr>
	<tr><th scope=row>94%</th><td>137.9672</td><td>138.8059</td><td>139.6474</td><td>140.4865</td><td>141.3249</td><td>142.1704</td><td>143.0139</td><td>143.8543</td><td>144.6879</td><td>145.5315</td><td>⋯</td><td>170.2367</td><td>171.2065</td><td>172.1768</td><td>173.1388</td><td>174.1076</td><td>175.0769</td><td>176.0446</td><td>177.0010</td><td>177.9730</td><td>178.9463</td></tr>
</tbody>
</table>



#### Prediction intervals

Now let's walk through generating and 89% prediction interval for actual heights, not just the average height, $\mu$. This means we'll incorporate the standard deviation $\sigma$ and its uncertainty as well. Remember, the first line of the statistical model here is
$$
h_i \sim \text{Normal}(\mu_i, \sigma)
$$

What we've done so far is just use samples from the posterior to visualize the uncertainty in $\mu_i$, the linear model of the mean. But actual predictions of the height also depend on the distribution in the first line.

Here's how we do that. Imagine simulating heights. For any unique weight value, you samples from a Gaussian distribution with the correct mean $\mu$ for that weight, using the correct value of $\sigma$ samples from the same posterior distribution. If we do this for every sample from the posterior, for every weight value of interest, we end up with a collection of simulated heights that embody the uncertainty in the posterior as well as the uncertainty in the Gaussian distribution of heights. There is a tool called `sim` which does this:


```R
sim.heights <- sim(m4.3, data=list(weight=weight.seq))
str(sim.heights)
```

     num [1:1000, 1:46] 130 130 133 146 144 ...


This is much like the earlier one, $\mu$, but contains simulated heights, not distributions of average heights.

We can summarize these simulated heights in the same way we summarize the distributions of $\mu$, by using `apply`:


```R
heights.PI <- apply(sim.heights, 2, PI, prob=0.89)
```

Now we can plot everything which we have:
1. The average line
1. The shaded region of 89% plausible $\mu$
1. The boundaries of the simulated heights the model expects


```R
# raw data
plot(height ~ weight, d2, col=col.alpha(rangi2, 0.5))

# MAP line
lines(weight.seq, mu.mean)

# Draw the HPDI for line
shade(mu.PI, weight.seq)

# Draw PI region for simulated heights
shade(heights.PI, weight.seq)
```

![png](sr-chapter4-output_90_0.png)

The roughness in the shaded region is due to variance in the simulation. We can always increase the number of points that we use to smooth it out:

```R
# increasing the number of points used
sim.heights <- sim(m4.3, data=list(weight=weight.seq), n=1e5)
heights.PI <- apply(sim.heights, 2, PI, prob=0.89)

plot(height ~ weight, d2, col=col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(heights.PI, weight.seq)
```


    
![png](sr-chapter4-output_92_0.png)
    


** Rolling your own `sim` **

Just like with `link`, it's useful to know how to do things.

For every distribution (like `dnorm`), there is a companion simulation function. For the Gaussian, that is `rnorm`, and it simulates sampling from a Gaussian distribution. What we want R to di is to simulate a height for eac hset of samples, and to do this for each value of weight. The following will do it:


```R
post <- extract.samples(m4.3)
weight.seq <- 25:70
sim.height <- sapply(weight.seq, function(weight)
    rnorm(
        n=nrow(post),
        mean=post$a + post$b * (weight - xbar),
        sd=post$sigma
    )
)
height.PI <- apply(sim.height, 2, PI, prob=0.89)
```

## Curves from lines

In the next chapter, we'll build linear models with more than one predictor variable. Before that though, let's build a model with a curve rather than a line. There are two common methods: _polynomial regression_ and _b-splines_.

### Polynomial regression
Polynomial regression uses powers of the predictor variable as additional predictors. This is an easy way to add some curviness to the models. Let's use polynomial regression to model the data for all of the !Kung, not just the adults.


```R
d <- Howell1

plot(height ~ weight, d)
```


    
![png](sr-chapter4-output_96_0.png)
    


The relationship is visibly curved - clearly a linear model will not work very well!

The most common polynomial refgression is a parabolic model of the mean. Let $x$ be the standardixed body weight. Then the parabolic equation for the mean height is:
$$
\mu_i = \alpha + \beta_1 x_i + \beta_2 x_i^2
$$

Fitting the model is easy - interpreting it is hard!

The first thing that we need to do is to *standardize* the predictor variable. This means applying the transformation $x \to \frac{x - \bar{x}}{sd(x)}$ - taking $x$ to its z score.

With this new parameter, here's our model:
$$
\begin{align*}
h_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_1 x_i + \beta_2 x_i ^ 2 \\
\alpha &\sim \text{Normal}(178, 20) \\
\beta_1 &\sim \text{Log-Normal}(0, 1) \\
\beta_2 &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Uniform}(0, 50) \\
\end{align*}
$$

The confusing thing here is assigning a prior for $\beta_2$, the parameter on the squared value of $x$.


```R
# building the model
d$weight_s <- (d$weight - mean(d$weight)) / sd(d$weight) # standardized
d$weight_s2 <- d$weight_s ^ 2 #squared - no need to calculate it every time

m4.5 <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu <- a + b1*weight_s + b2*weight_s2,
        a ~ dnorm(178, 20),
        b1 ~ dlnorm(0, 1),
        b2 ~ dnorm(0, 1),
        sigma ~ dunif(0, 50)
    ),
    data=d
)
```


```R
precis(m4.5)
```


<table class="dataframe">
<caption>A precis: 4 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>146.055481</td><td>0.3689255</td><td>145.465867</td><td>146.645095</td></tr>
	<tr><th scope=row>b1</th><td> 21.733026</td><td>0.2888405</td><td> 21.271403</td><td> 22.194649</td></tr>
	<tr><th scope=row>b2</th><td> -7.801266</td><td>0.2741548</td><td> -8.239418</td><td> -7.363113</td></tr>
	<tr><th scope=row>sigma</th><td>  5.773496</td><td>0.1763920</td><td>  5.491587</td><td>  6.055404</td></tr>
</tbody>
</table>



It's a little more difficult to interpret this, so let's plot it instead!


```R
weight.seq <- seq(from=-2.2, to=2.2, length.out=30)
pred_dat <- list(weight_s=weight.seq, weight_s2=weight.seq^2)
mu <- link(m4.5, data=pred_dat)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob=0.89)
sim.height <- sim(m4.5, data=pred_dat)
height.PI <- apply(sim.height, 2, PI, prob=0.89)
```


```R
plot(height ~ weight_s, d, col=col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)
```


    
![png](sr-chapter4-output_102_0.png)
    


This looks like a better approximation than the linear one, especially at the ends! We might be able to make it better by using another, cubic term:
$$
\begin{align*}
h_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_1 x_i + \beta_2 x_i ^ 2 + \beta_3 x_i ^ 3 \\
\alpha &\sim \text{Normal}(178, 20) \\
\beta_1 &\sim \text{Log-Normal}(0, 1) \\
\beta_2 &\sim \text{Normal}(0, 1) \\
\beta_3 &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Uniform}(0, 50) \\
\end{align*}
$$


```R
d$weight_s3 <- d$weight_s^3
m4.6 <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu <- a + b1*weight_s + b2*weight_s2 + b3*weight_s3,
        a ~ dnorm(178, 20),
        b1 ~ dlnorm(0, 1),
        b2 ~ dnorm(0, 1),
        b3 ~ dnorm(0, 1),
        sigma ~ dunif(0, 50)
    ),
    data=d
)
```


```R
weight.seq <- seq(from=-2.2, to=2.2, length.out=30)
pred_dat <- list(weight_s=weight.seq, weight_s2=weight.seq^2, weight_s3=weight.seq^3)
mu <- link(m4.6, data=pred_dat)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob=0.89)
sim.height <- sim(m4.6, data=pred_dat)
height.PI <- apply(sim.height, 2, PI, prob=0.89)

plot(height ~ weight_s, d, col=col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)
```

![png](sr-chapter4-output_105_0.png)

This looks even better! One of the problems is that this model doesn't really give us any sort of insight into the process here - why would a cubic model fit human growth curves? Who knows! Also we need to be careful - just because a model fits the data better doesn't mean that it is a better model. Also, we can always just keep adding terms to fit the data perfectly - obviously this isn't the best idea!

If we want to, we can also recover the initial units (rather than the z-scores):


```R
plot(height ~ weight_s, d, col=col.alpha(rangi2, 0.5), xaxt="n") # turn off the axis

# now we explicity construct the axis:
at <- c(-2, 1, 0, 1, 2)
labels <- at * sd(d$weight) + mean(d$weight)
axis(side=1, at=at, labels=round(labels,1))
```


![png](sr-chapter4-output_107_0.png)

### Splines

A spline was originally a piece of wood or whatnot that was fixed in a few places and allowed to assume a smooth curve. Mathematically, a spline is a smooth function which is built out of smaller component functions. There are many kinds of splines. Here, we'll look at *b-splines* - the "b" stands for "basis".

We'll use cherry blossom data to see how they work.


```R
library(rethinking)
data(cherry_blossoms)
d <- cherry_blossoms
precis(d)
```


<table class="dataframe">
<caption>A precis: 5 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>year</th><td>1408.000000</td><td>350.8845964</td><td>867.77000</td><td>1948.23000</td><td>▇▇▇▇▇▇▇▇▇▇▇▇▁  </td></tr>
	<tr><th scope=row>doy</th><td> 104.540508</td><td>  6.4070362</td><td> 94.43000</td><td> 115.00000</td><td>▁▂▅▇▇▃▁▁       </td></tr>
	<tr><th scope=row>temp</th><td>   6.141886</td><td>  0.6636479</td><td>  5.15000</td><td>   7.29470</td><td>▁▃▅▇▃▂▁▁       </td></tr>
	<tr><th scope=row>temp_upper</th><td>   7.185151</td><td>  0.9929206</td><td>  5.89765</td><td>   8.90235</td><td>▁▂▅▇▇▅▂▂▁▁▁▁▁▁▁</td></tr>
	<tr><th scope=row>temp_lower</th><td>   5.098941</td><td>  0.8503496</td><td>  3.78765</td><td>   6.37000</td><td>▁▁▁▁▁▁▁▃▅▇▃▂▁▁▁</td></tr>
</tbody>
</table>




```R
plot(doy ~ year, data=d)
```


    
![png](sr-chapter4-output_110_0.png)
    


There might be some sort of wiggly trend here - it's hard to tell!

Let's try to build up a prediction by using splines. Basically, we are going to divide up `year` into lots of little segments, and assign a parameter to each part. The parameters are turned on and off appropriately. Each of the synthetic variables created by each spline component is there to turn some stuff on and off. Each of the synthetic variables is called a *basis function*. Then the linear model ends up being
$$
\mu_i = \alpha + w_1 B_{i,1} + w_2 B_{i,2} + w_3 B_{i,3} + \dots
$$

Where $B_{i,n}$ is the value of the $n$th basis function on row $i$, and the $w$ parameters are the weights for each. The parameters act like slopes, adjusting the influence of each basis function onf the mean $\mu_i$.

How do we construct the basis variables $B$? First we divide the full range on the horizontal axis into four parts, using pivot points called *knots*. The knots act as pivots for five different basis functions, our $B$ variables. The variables are used to gently transition from onie region to the next. Beginning at the left, basis function 1 has value 1 and all of the others are set to zero. As we move rightwards towards the second know, basis 1 declines and bases 2 increates. At know 2, basis 2 has value 1, and all of the others are set to zero.

One nice thing is that the basis functions are very local. At each point on the horizontal axis, only two (at most) basis functions have a non-zero value - the ones on either side of the point.

So, let's build the code. First we need to get the knots. Let's evenly space them by quantiles.


```R
d2 <- d[complete.cases(d$doy),]
num_knots <- 15
knot_list <- quantile(d2$year, probs=seq(0,1,length.out=num_knots))
knot_list
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>0%</dt><dd>812</dd><dt>7.142857%</dt><dd>1036</dd><dt>14.28571%</dt><dd>1174</dd><dt>21.42857%</dt><dd>1269</dd><dt>28.57143%</dt><dd>1377</dd><dt>35.71429%</dt><dd>1454</dd><dt>42.85714%</dt><dd>1518</dd><dt>50%</dt><dd>1583</dd><dt>57.14286%</dt><dd>1650</dd><dt>64.28571%</dt><dd>1714</dd><dt>71.42857%</dt><dd>1774</dd><dt>78.57143%</dt><dd>1833</dd><dt>85.71429%</dt><dd>1893</dd><dt>92.85714%</dt><dd>1956</dd><dt>100%</dt><dd>2015</dd></dl>



The next choice is the degree of the polynomial. This determines how basis functions combine, which determines how the parameters interact to produce the spline. Luckily, R has a function that can do all of this for us


```R
library(splines)
B <- bs(d2$year,
       knots=knot_list[-c(1,num_knots)],
        degree=3,
        intercept=TRUE
   )
B
```


<table class="dataframe">
<caption>A bs: 827 × 17 of type dbl</caption>
<thead>
	<tr><th scope=col>1</th><th scope=col>2</th><th scope=col>3</th><th scope=col>4</th><th scope=col>5</th><th scope=col>6</th><th scope=col>7</th><th scope=col>8</th><th scope=col>9</th><th scope=col>10</th><th scope=col>11</th><th scope=col>12</th><th scope=col>13</th><th scope=col>14</th><th scope=col>15</th><th scope=col>16</th><th scope=col>17</th></tr>
</thead>
<tbody>
	<tr><td>1.00000000</td><td>0.0000000</td><td>0.0000000000</td><td>0.000000e+00</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.96035713</td><td>0.0393123</td><td>0.0003298367</td><td>7.286030e-07</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.76650948</td><td>0.2207460</td><td>0.0125594810</td><td>1.850922e-04</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.56334070</td><td>0.3856737</td><td>0.0493848352</td><td>1.600741e-03</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.54526700</td><td>0.3986837</td><td>0.0541894688</td><td>1.859854e-03</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.45273210</td><td>0.4597597</td><td>0.0837138624</td><td>3.794349e-03</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.43712204</td><td>0.4690287</td><td>0.0896000902</td><td>4.249213e-03</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.41438627</td><td>0.4819157</td><td>0.0987005024</td><td>4.997488e-03</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.28262329</td><td>0.5387095</td><td>0.1663475144</td><td>1.231968e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.27124388</td><td>0.5417994</td><td>0.1736519200</td><td>1.330480e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.26567055</td><td>0.5431801</td><td>0.1773329095</td><td>1.381647e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.25475398</td><td>0.5456182</td><td>0.1847489784</td><td>1.487883e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.24940967</td><td>0.5466778</td><td>0.1884826646</td><td>1.542984e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.24414062</td><td>0.5476326</td><td>0.1922325232</td><td>1.599429e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.21407716</td><td>0.5512183</td><td>0.2150322805</td><td>1.967228e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.18658892</td><td>0.5512975</td><td>0.2382386867</td><td>2.387486e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.16963716</td><td>0.5495163</td><td>0.2538612315</td><td>2.698530e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.16556605</td><td>0.5488515</td><td>0.2577794327</td><td>2.780298e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.14993286</td><td>0.5453467</td><td>0.2734815984</td><td>3.123886e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.12837820</td><td>0.5376696</td><td>0.2970462987</td><td>3.690593e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.11842244</td><td>0.5328046</td><td>0.3087930484</td><td>3.997991e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.10596771</td><td>0.5253184</td><td>0.3243761995</td><td>4.433771e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.09722269</td><td>0.5189914</td><td>0.3359797939</td><td>4.780610e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.07628282</td><td>0.4993697</td><td>0.3664184188</td><td>5.792904e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.05858868</td><td>0.4761814</td><td>0.3958411535</td><td>6.938873e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.05659151</td><td>0.4730594</td><td>0.3994298012</td><td>7.091930e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.04728365</td><td>0.4567788</td><td>0.4170269604</td><td>7.891060e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.04386693</td><td>0.4499737</td><td>0.4238913166</td><td>8.226805e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.04222209</td><td>0.4465127</td><td>0.4272833137</td><td>8.398192e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0.04061890</td><td>0.4430141</td><td>0.4306475945</td><td>8.571942e-02</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.861705e-02</td><td>0.2466951637</td><td>0.60322352</td><td>0.1314643</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.675679e-02</td><td>0.2333123635</td><td>0.60487707</td><td>0.1450538</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.502478e-02</td><td>0.2200479874</td><td>0.60537830</td><td>0.1595489</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.341643e-02</td><td>0.2069275761</td><td>0.60467706</td><td>0.1749789</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.192715e-02</td><td>0.1939766703</td><td>0.60272315</td><td>0.1913730</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.055238e-02</td><td>0.1812208108</td><td>0.59946642</td><td>0.2087604</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>9.287531e-03</td><td>0.1686855383</td><td>0.59485667</td><td>0.2271703</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>8.128021e-03</td><td>0.1563963935</td><td>0.58884375</td><td>0.2466318</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>7.069271e-03</td><td>0.1443789173</td><td>0.58137747</td><td>0.2671743</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>6.106702e-03</td><td>0.1326586503</td><td>0.57240765</td><td>0.2888270</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>5.235734e-03</td><td>0.1212611334</td><td>0.56188413</td><td>0.3116190</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4.451786e-03</td><td>0.1102119071</td><td>0.54975672</td><td>0.3355796</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3.750279e-03</td><td>0.0995365124</td><td>0.53597526</td><td>0.3607380</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3.126632e-03</td><td>0.0892604899</td><td>0.52048956</td><td>0.3871233</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2.576265e-03</td><td>0.0794093803</td><td>0.50324946</td><td>0.4147649</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2.094599e-03</td><td>0.0700087245</td><td>0.48420477</td><td>0.4436919</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.677053e-03</td><td>0.0610840632</td><td>0.46330533</td><td>0.4739336</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.319048e-03</td><td>0.0526609370</td><td>0.44050095</td><td>0.5055191</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.016003e-03</td><td>0.0447648868</td><td>0.41574147</td><td>0.5384776</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>7.633378e-04</td><td>0.0374214533</td><td>0.38897670</td><td>0.5728385</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>5.564733e-04</td><td>0.0306561772</td><td>0.36015648</td><td>0.6086309</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3.908290e-04</td><td>0.0244945993</td><td>0.32923062</td><td>0.6458840</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2.618249e-04</td><td>0.0189622603</td><td>0.29614896</td><td>0.6846270</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1.648810e-04</td><td>0.0140847010</td><td>0.26086131</td><td>0.7248891</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>9.541723e-05</td><td>0.0098874622</td><td>0.22331751</td><td>0.7666996</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4.885362e-05</td><td>0.0063960844</td><td>0.18346737</td><td>0.8100877</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2.061012e-05</td><td>0.0036361086</td><td>0.14126073</td><td>0.8550826</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>6.106702e-06</td><td>0.0016330754</td><td>0.09664740</td><td>0.9017134</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>7.633378e-07</td><td>0.0004125256</td><td>0.04957722</td><td>0.9500095</td></tr>
	<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.000000e+00</td><td>0.0000000000</td><td>0.00000000</td><td>1.0000000</td></tr>
</tbody>
</table>



The matrix `B` has 827 rows and 17 columns. Each row is a yearm corresponding to the rows in the `d2` dataframe. Each colum is a basis function, one of our synthetic variables defining a span of years within which a corresponding parameter will influence prediction. To display the basis functions, just plot each column against the year:


```R
library(repr)
options(repr.plot.width=14)
plot(NULL, xlim=range(d2$year), ylim=c(0,1), xlab="year", ylab="basis")
for (i in 1:ncol(B)) {
    lines(d2$year, B[,i])
}
```

![png](sr-chapter4-output_116_0.png)

To get the actual weights, we need to define the model and make it run! In mathemtical terms:
$$
\begin{align*}
D_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha + \sum_{k=1}^K w_k B_{k,i} \\
\end{align*}
$$

And then the priors:
$$
\begin{align*}
\alpha &\sim \text{Normal}(100,10) \\
w_j &\sim \text{Normal}(0,10) \\
\sigma &\sim \text{Exponential}(1)
\end{align*}
$$

This is also the first time that we've used an exponential distribution. They are useful for scale parameters - ones that must be positive. The way to read an exponential distribution is to think of it as containing no more information than an average deviation. The average is the inverse of the rate. So in this case it is $1 / 1 = 1$.

Now to the model. The only trick is how to do the sum, and for that we just use matrix multiplication.


```R
m4.7 <- quap(
    alist(
        D ~ dnorm(mu, sigma),
        mu <- a + B %*% w,
        a ~ dnorm(100, 10),
        w ~ dnorm(0, 10),
        sigma ~ dexp(1)
    ),
    data=list(D=d2$doy, B=B),
    start=list(w=rep(0,ncol(B))),
)
```


```R
# Let's plot the posterior predictions to see what is going on
post <- extract.samples(m4.7)
w <- apply(post$w, 2, mean)
plot(NULL, xlim=range(d2$year), ylim=c(-6,6), xlab="year", ylab="basis * weight")
for (i in 1:ncol(B)) lines(d2$year, w[i]*B[,i])
```


    
![png](output_120_0.png)
    



```R
# 97% posterior interval for mu
mu <- link(m4.7)
mu_PI <- apply(mu, 2, PI, 0.97)
plot(d2$year, d2$doy, col=col.alpha(rangi2, 0.3), pch=16)
shade(mu_PI, d2$year, col=col.alpha("black", 0.5))
```


    
![png](sr-chapter4-output_121_0.png)
    


### Smooth functions for a rough world

The splines in the previous section are just the beginning. An entire class of models, *generalized additive models* (GAMs), focuses on predicting an outcome variable using smooth functions of some predictor variables. The topic is deep enough to deserve its own book.

## Practice

*4E1* In the model definition below, which line is the likelihood?
$$
\begin{align}
y_i &\sim \text{Normal}(\mu, \sigma) \\
\mu &\sim \text{Normal}(0, 10) \\
\sigma &\sim \text{Exponential}(1)
\end{align}
$$

The first one.

*4E2* In the model definition just above, how many parameters are in the posterior distribution?

Two - $\mu$ and $\sigma$.

*4E3* Using the model definition above, write down the appropriate form of Bayes' theorem that includes the proper likelihood and priors

Bayes' Theorem: $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$

$$
\begin{align*}
Pr(\mu,\sigma | y) &= \frac{Pr(y | \mu, \sigma) Pr(\mu, \sigma)}{\int \int Pr(y|\mu,\sigma) Pr(\sigma) \Pr(\mu) d\mu d\sigma} \\
                    &= \frac{Normal(y|\mu, \sigma) Normal(\mu|0,10) Exponential(\sigma|1)}{\int \int Normal(y|\mu, \sigma) Normal(\mu|0,10) Exponential(\sigma|1)d\mu d\sigma}
\end{align*}
$$

*4E4* In the model definition below, which line is the linear model?
$$
\begin{align*}
y_i &\sim \text{Normal}(\mu,\sigma) \\
\mu_i &= \alpha + \beta x_i \\
\alpha &\sim \text{Normal}(0,10) \\
\beta &\sim \text{Normal}(0,1) \\
\sigma &\sim \text{Exponential}(2)
\end{align*}
$$

The second line

*4E5* In the model definition just above, how many parameters are in the posterior distribution?

Three - $\alpha$, $\beta$, and $\sigma$.

*4M1* For the model definition below, simulate observed $y$ values from the prior.
$$
\begin{align*}
y_i &\sim \text{Normal}(\mu,\sigma) \\
\mu &\sim \text{Normal}(0,10) \\
\sigma &\sim \text{Exponential}(1) \\
\end{align*}
$$


```R
NUM_SAMPLES = 1E4
mus = rnorm(NUM_SAMPLES, 0, 10)
sigmas = rexp(NUM_SAMPLES, 1)
ys = rnorm(NUM_SAMPLES, mus, sigmas)
dens(ys)
```


    
![png](sr-chapter4-output_129_0.png)
    


*4M2* Translate the above into a `quap` formula


```R
alist(
        y ~ dnorm(mu, sigma),
        mu ~ dnorm(0,10),
        sigma ~ dexp(1)
    )
```


    [[1]]
    y ~ dnorm(mu, sigma)
    
    [[2]]
    mu ~ dnorm(0, 10)
    
    [[3]]
    sigma ~ dexp(1)



*4M3* Translate the `quap` model formula below into a mathematical model definition

```R
y ~ dnorm(mu, sigma),
mu <- a + b * x,
a ~ dnorm(0,10),
b ~ dunif(0,1),
sigma ~ dexp(1)
```

$$
\begin{align*}
y_i &\sim \text{Normal}(\mu_i,\sigma) \\
\mu_i &= a + b*x_i \\
a &\sim \text{Normal}(0,10) \\
b &\sim \text{Uniform}(0,1) \\
\sigma &\sim \text{Exponential}(1) \\
\end{align*}
$$

*4M4* A samples of students is measured for heights each year for 3 years. After the third year, you want to fit a linear regression predicting height using year as a predictor. Write down the matehmatical model definition for this regression, using any variable names and priors you choose. Be prepared to defend your choice of priors.

$$
\begin{align*}
h_i &\sim Normal(\mu_i,\sigma) \\
\mu_i &= \alpha + \beta y_i \\
\alpha &\sim Normal(176, 20) \\
\beta &\sim Normal(1, 2) \\
\sigma &\sim Log-Normal(0, 20) \\
\end{align*}
$$

*4M5* Now suppose that I remind you that every student got taller every year. Does this information lead you to change your choice of priors? How?

Yes - need to ensure that the parameter $\beta$ is positive:
$$
\begin{align*}
h_i &\sim Normal(\mu_i,\sigma) \\
\mu_i &= \alpha + \beta y_i \\
\alpha &\sim Log-Normal(0, 1) \\
\beta &\sim Normal(1, 2) \\
\sigma &\sim Log-Normal(0, 1) \\
\end{align*}
$$

*4M6* Now suppose that I tell you that the variance among heights for students of the same age is never more than 64 cm. Does this information lead you to revise your priors?

Yes - need to ensure that $\sigma \in [0, 64]$. Probably the best way to do this is to ensure that it comes from a uniform distribution.
$$
\begin{align*}
h_i &\sim Normal(\mu_i,\sigma) \\
\mu_i &= \alpha + \beta y_i \\
\alpha &\sim Log-Normal(0, 1) \\
\beta &\sim Normal(1, 2) \\
\sigma &\sim Uniform(0, 64) \\
\end{align*}
$$

*4M7* Refit model `m4.43` from the chapter, but omit the mean `xbar`. Compare the new model's posterior to that of the original model. In particular, look at the covariance among the parameters. What is different? Then compare the posterior predictions of both models.


```R
d <- Howell1
d2 <- d[d$age >= 18,]
m4m7 <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu <- a + b * weight,
        a ~ dnorm(178, 20),
        b ~ dlnorm(0, 1),
        sigma ~ dunif(0, 50)
    ),
    data=d2,
)
```


```R
round(vcov(m4.3),3)
```


<table class="dataframe">
<caption>A matrix: 3 × 3 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>a</th><th scope=col>b</th><th scope=col>sigma</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>0.073</td><td>0.000</td><td>0.000</td></tr>
	<tr><th scope=row>b</th><td>0.000</td><td>0.002</td><td>0.000</td></tr>
	<tr><th scope=row>sigma</th><td>0.000</td><td>0.000</td><td>0.037</td></tr>
</tbody>
</table>




```R
round(vcov(m4m7), 3)
```


<table class="dataframe">
<caption>A matrix: 3 × 3 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>a</th><th scope=col>b</th><th scope=col>sigma</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td> 3.601</td><td>-0.078</td><td>0.009</td></tr>
	<tr><th scope=row>b</th><td>-0.078</td><td> 0.002</td><td>0.000</td></tr>
	<tr><th scope=row>sigma</th><td> 0.009</td><td> 0.000</td><td>0.037</td></tr>
</tbody>
</table>



So it looks here like there is now a much larger covariance among the different parameters.


```R
for (model in c(m4.3, m4m7)) {
    sim.heights <- sim(model, data=list(weight=weight.seq), n=1e4)
    heights.PI <- apply(sim.heights, 2, PI, prob=0.89)

    plot(height ~ weight, d2, col=col.alpha(rangi2, 0.5))
    lines(weight.seq, mu.mean)
    shade(mu.PI, weight.seq)
    shade(heights.PI, weight.seq)
}
```


    
![png](sr-chapter4-output_141_0.png)
    



    
![png](sr-chapter4-output_141_1.png)
    


These predictions look to be the same!

*4M8* In the chapter, we used 15 knots with the cherry blossom spline. INcrease the number of knots and observe what happens to the resulting spline. Then adjust also the width of the prior on the weights - change the standard deviation of the prior and watch what happens. What do you think the comination of know k=number and the prior on the weights controls?


```R
d <- cherry_blossoms
d2 <- d[complete.cases(d$doy),]
for (deviation in c(1, 2, 3)) {
    for (num_knots in c(1, 5, 15, 45)) {
        knot_list <- quantile(d2$year, probs=seq(0,1,length.out=num_knots))
        B <- bs(d2$year,
               knots=knot_list[-c(1,num_knots)],
                degree=3,
                intercept=TRUE
           )
        m4m8 <- quap(
            alist(
                D ~ dnorm(mu, sigma),
                mu <- a + B %*% w,
                a ~ dnorm(100, 10),
                w ~ dnorm(0, 10),
                sigma ~ dexp(deviation)
            ),
            data=list(D=d2$doy, B=B),
            start=list(w=rep(0,ncol(B))),
        )
        post <- extract.samples(m4m8)
        w <- apply(post$w, 2, mean)
        mu <- link(m4m8)
        mu_PI <- apply(mu, 2, PI, 0.97)
        plot(d2$year, d2$doy, col=col.alpha(rangi2, 0.3), pch=16, main=concat("Number of knots: ", num_knots, "; Prior SD: ", deviation))
        shade(mu_PI, d2$year, col=col.alpha("black", 0.5))
    }
}

```

![png](sr-chapter4-output_144_0.png)
    



    
![png](sr-chapter4-output_144_1.png)
    



    
![png](sr-chapter4-output_144_2.png)
    



    
![png](sr-chapter4-output_144_3.png)
    



    
![png](sr-chapter4-output_144_4.png)
    



    
![png](sr-chapter4-output_144_5.png)
    



    
![png](sr-chapter4-output_144_6.png)
    



    
![png](sr-chapter4-output_144_7.png)
    



    
![png](sr-chapter4-output_144_8.png)
    



![png](sr-chapter4-output_144_9.png)

![png](sr-chapter4-output_144_10.png)

![png](sr-chapter4-output_144_11.png)

So, changing the number of knots seems to affect the wiggliness of the result, although the prior SD doesn't seem to have that much of an effect.

*4H1* The weights listed below are recorded in the !Kung census, but heights are not recorded for the individuals. Provide predicted heights and 89% intervals for each of these individuals. That is, fill in the table below, using model-based predictions.

|Individual | Weight | Expected height | 89% interval |
| --- | --- | --- | --- |
| 1 | 46.95 |
| 2 | 43.72 |
| 3 | 64.78 |
| 4 | 32.59 |
| 5 | 54.63 |


```R
num_samples <- 1e4
post <- extract.samples(m4.3)
for (weight in c(46.95, 43.72, 64.78, 32.59, 54.63)) {
    mus <- post$a + post$b * (weight - xbar)
    heights <- rnorm(num_samples, mus, post$sigma)
    print(concat("Weight: ", weight, "; expected height: ", mean(heights)))
    print(PI(heights, prob=0.89))
}
```

    [1] "Weight: 46.95; expected height: 156.317665010796"
          5%      94% 
    148.0900 164.5259 
    [1] "Weight: 43.72; expected height: 153.49619798218"
          5%      94% 
    145.6186 161.7106 
    [1] "Weight: 64.78; expected height: 172.395778278508"
          5%      94% 
    163.9979 180.7353 
    [1] "Weight: 32.59; expected height: 143.379062712885"
          5%      94% 
    135.1722 151.5093 
    [1] "Weight: 54.63; expected height: 163.323431308543"
          5%      94% 
    155.1950 171.4784 


*4H2* Select out all of the rows in the `Howell1` data with ages below 18. If you do it right, you should end up with a new data frame with 192 rows in it
a) Fit a linear regression to these data, using `quap`. Present and interpret the estimates. For every 10 units of increase in weight, how much taller does the model predict the child gets?
b) Plot the raw data, with height on the vertical axis and weight on the horizontal axis. Superimpose the MAP regression line and 89% interval for the mean. Also superimpose the 89% interval for the predicted heights.
c) What aspects of the model concern you? Describe the kinds of assumptions you would change, if any, to improve the mode. You don't have to write any new code. Just explain what the model appears to be doing a bad job of, and what you hypothesize would be a better model.


```R
# a)
d <- Howell1
d4h2 <- d[d$age < 18,]
head(d4h2)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>height</th><th scope=col>weight</th><th scope=col>age</th><th scope=col>male</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>19</th><td>121.92</td><td>19.61785</td><td>12.0</td><td>1</td></tr>
	<tr><th scope=row>20</th><td>105.41</td><td>13.94795</td><td> 8.0</td><td>0</td></tr>
	<tr><th scope=row>21</th><td> 86.36</td><td>10.48931</td><td> 6.5</td><td>0</td></tr>
	<tr><th scope=row>24</th><td>129.54</td><td>23.58678</td><td>13.0</td><td>1</td></tr>
	<tr><th scope=row>25</th><td>109.22</td><td>15.98912</td><td> 7.0</td><td>0</td></tr>
	<tr><th scope=row>29</th><td>137.16</td><td>27.32892</td><td>17.0</td><td>1</td></tr>
</tbody>
</table>




```R
nrow(d4h2)
```


192



```R
wbar <- mean(d4h2$weight)
model <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu <- a + b * weight,
        a ~ dnorm(100, 20),
        b ~ dlnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data=d4h2
)
```


```R
precis(model)
```


<table class="dataframe">
<caption>A precis: 3 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>58.485593</td><td>1.36557687</td><td>56.303137</td><td>60.668048</td></tr>
	<tr><th scope=row>b</th><td> 2.708246</td><td>0.06677885</td><td> 2.601520</td><td> 2.814971</td></tr>
	<tr><th scope=row>sigma</th><td> 8.261958</td><td>0.40870468</td><td> 7.608768</td><td> 8.915147</td></tr>
</tbody>
</table>



This indicates that for each 10 units increase in weight, we would expect to see a roughly 27cm increase in height.


```R
# b)
post <- extract.samples(model)
weight_seq <- seq(from=1, to=45, length.out=50)
mu <- sapply(weight_seq, function(weight) {
    mean(post$a + post$b * weight)
})

ci <- sapply(weight_seq, function(weight) {
    PI(post$a + post$b * weight, prob=0.89)
})

sim_heights <- sim(model, data=list(weight=weight_seq))
sim_heights_ci <- apply(sim_heights, 2, PI, prob=0.89)

plot(height ~ weight, data=d4h2, main="Height vs. Weight for Children")
lines(weight_seq, mu)
shade(ci, weight_seq)
shade(sim_heights_ci, weight_seq)
```


    
![png](sr-chapter4-output_154_0.png)
    


c) This model seems to do well in the middle~ of the interval, but systematically overestimates heights to the right (heavier children) and left (lighter children). There is definitely a curve to the data, so some sort of curved model (e.g. quadratic regression) might fit the data better.

*4H3* Suppose that a colleague of yours, who works on allometry, glances at the practice problem up above. Your colleague exclaims, "That's silly! Everyone knows that it's only the logarithm of body weight that scales with height!" let's take the colleague's advice and see what happens.
a) Model the relationship between height and the natural logarithm of weight. Use the entire `Howell1` data set - all 544 rows of adults and children. Can you interpret the resulting estimates?
b) Begin with this plot: `plot(height ~ weight, data=Howell1)`. Then use samples from the quadratic approcimate posterior of the model in a) to superimpose on the plot:
1. The predicted mean height as a function of weight;
1. The 97% interval for the mean;
1. The 97% interval for predicted heights


```R
# a)
d <- Howell1
d4h3 <- d

model <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu <- a + b * log(weight),
        a ~ dnorm(100, 10),
        b ~ dlnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data=d4h3
)
```


```R
precis(model)
```


<table class="dataframe">
<caption>A precis: 3 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>-21.611652</td><td>1.3273670</td><td>-23.733041</td><td>-19.490263</td></tr>
	<tr><th scope=row>b</th><td> 46.460955</td><td>0.3803681</td><td> 45.853053</td><td> 47.068856</td></tr>
	<tr><th scope=row>sigma</th><td>  5.123559</td><td>0.1550072</td><td>  4.875828</td><td>  5.371291</td></tr>
</tbody>
</table>




```R
# b)
plot(height ~ weight, data=Howell1)

weight_seq <- seq(from=1.4, to=4.2, length.out=50)
post <- extract.samples(model)

mean_line <- sapply(weight_seq, function(weight) {
    mean(post$a + post$b * weight)
})

heights <- sapply(weight_seq, function(weight) {
    PI(
        rnorm(1000, post$a + post$b * weight, post$sigma),
        prob=0.97
    )
})

lines(exp(weight_seq), mean_line)
shade(heights, exp(weight_seq))
```

![png](sr-chapter4-output_159_0.png)
