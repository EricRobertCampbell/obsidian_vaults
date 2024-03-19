[[Generalized Linear Model]]s are quite a bit more complicated than the standard linear model - it can be hard to figure out exactly how the parameters will affect the prediction.

Most common GLMs are for counts. The basic issue is that the scale of the parameters (continuous, linear, &c.) is not the same as the scale of the output (discrete).

In this chapter we'll look at two kinds of [[Generalized Linear Model|GLMs]]:
1. [[Binomial regression]] consists of a family of related models for binary classification (right / wrong, left / right, accept / reject, &c.) for which the total is known
1. [[Poisson regression]] is a [[Generalized Linear Model|GLM]] for modelling a count with an unknown maximum (e.g. number of elephants in Kenya)

At the end of the chapter we'll look at some other related count models.

## 1 Binomial Regression

Think back to the globe tossing model of the [[SR - Chapter 2 - Small Worlds and Large|earlier chapter]]. The model was binomial - it counted the number of water samples when tossing the globe. It wasn't a GLM because there were no predictor variables.

There are two different GLMs that use a binomial probability function, and really they are the same model, just organizing the data differently.
1. [[Logistic Regression]] is used when the data are organized into single-trial cases, such that the outcome variable can only take the values of 0 or 1
1. [[Aggregated Binomial Regression]] are used when the individual trials with the same covariate values are grouped together, so the outcome can take the value 0 - $n$, the number of trials.

Both use the **logit link** and so are sometimes both called logistic regression.

### 1.1 Logistic regression: Prosocial chimpanzees

The data for this example come from a study of the prosocial behaviour of chimpanzees (*Pan troglodytes*).

We will test to see if there is a connection between the option to deliver two pieces of food and the presence of another chimp.

```R
library(rethinking)
library(ggplot2)
options(repr.plot.width=16, repr.plot.height=8)
library(dagitty)
```
```R
data(chimpanzees)
d <- chimpanzees
```
```R
head(d)
summary(d)
```

<table class="dataframe">
<caption>A data.frame: 6 × 8</caption>
<thead>
	<tr><th></th><th scope=col>actor</th><th scope=col>recipient</th><th scope=col>condition</th><th scope=col>block</th><th scope=col>trial</th><th scope=col>prosoc_left</th><th scope=col>chose_prosoc</th><th scope=col>pulled_left</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>NA</td><td>0</td><td>1</td><td> 2</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>NA</td><td>0</td><td>1</td><td> 4</td><td>0</td><td>0</td><td>1</td></tr>
	<tr><th scope=row>3</th><td>1</td><td>NA</td><td>0</td><td>1</td><td> 6</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>1</td><td>NA</td><td>0</td><td>1</td><td> 8</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><th scope=row>5</th><td>1</td><td>NA</td><td>0</td><td>1</td><td>10</td><td>1</td><td>1</td><td>1</td></tr>
	<tr><th scope=row>6</th><td>1</td><td>NA</td><td>0</td><td>1</td><td>12</td><td>1</td><td>1</td><td>1</td></tr>
</tbody>
</table>


         actor     recipient     condition       block         trial      
     Min.   :1   Min.   :2     Min.   :0.0   Min.   :1.0   Min.   : 1.00  
     1st Qu.:2   1st Qu.:3     1st Qu.:0.0   1st Qu.:2.0   1st Qu.:18.75  
     Median :4   Median :5     Median :0.5   Median :3.5   Median :36.50  
     Mean   :4   Mean   :5     Mean   :0.5   Mean   :3.5   Mean   :36.50  
     3rd Qu.:6   3rd Qu.:7     3rd Qu.:1.0   3rd Qu.:5.0   3rd Qu.:54.25  
     Max.   :7   Max.   :8     Max.   :1.0   Max.   :6.0   Max.   :72.00  
                 NA's   :252                                              
      prosoc_left   chose_prosoc     pulled_left    
     Min.   :0.0   Min.   :0.0000   Min.   :0.0000  
     1st Qu.:0.0   1st Qu.:0.0000   1st Qu.:0.0000  
     Median :0.5   Median :1.0000   Median :1.0000  
     Mean   :0.5   Mean   :0.5675   Mean   :0.5794  
     3rd Qu.:1.0   3rd Qu.:1.0000   3rd Qu.:1.0000  
     Max.   :1.0   Max.   :1.0000   Max.   :1.0000  
                                                    


```R
?chimpanzees
```

    chimpanzees             package:rethinking             R Documentation
    
    _C_h_i_m_p_a_n_z_e_e _p_r_o_s_o_c_i_a_l_t_y _e_x_p_e_r_i_m_e_n_t _d_a_t_a
    
    _D_e_s_c_r_i_p_t_i_o_n:
    
         Data from behavior trials in a captive group of chimpanzees,
         housed in Lousiana. From Silk et al. 2005. Nature 437:1357-1359.
    
    _U_s_a_g_e:
    
         data(chimpanzees)
         
    _F_o_r_m_a_t:
    
           1. actor : name of actor
    
           2. recipient : name of recipient (NA for partner absent
              condition)
    
           3. condition : partner absent (0), partner present (1)
    
           4. block : block of trials (each actor x each recipient 1 time)
    
           5. trial : trial number (by chimp = ordinal sequence of trials
              for each chimp, ranges from 1-72; partner present trials were
              interspersed with partner absent trials)
    
           6. prosocial_left : 1 if prosocial (1/1) option was on left
    
           7. chose_prosoc : choice chimp made (0 = 1/0 option, 1 = 1/1
              option)
    
           8. pulled_left : which side did chimp pull (1 = left, 0 = right)
    
    _A_u_t_h_o_r(_s):
    
         Richard McElreath
    
    _R_e_f_e_r_e_n_c_e_s:
    
         Silk et al. 2005. Nature 437:1357-1359.


Variables to focus on:
- `pulled_left` as the outcome - this is whether they pulled the left hand lever (0 / 1)
- `prosoc_left` as a predictor - this is whether the left lever was attached to the option to deliver two pieces of food (0 / 1)
- `condition` as a predictor - this is whether there was a second chimp there (0 - no, 1 - yes)

We especially want to consider the four options:
1. `prosoc_left=0` and `condition=0`: two food items on the right and no partner
1. `prosoc_left=1` and `condition=0`: two food items on the left and no partner
1. `prosoc_left=0` and `condition=1`: two food items on the right and partner present
1. `prosoc_left=1` and `condition=1`: two food items on the right and partner present

The conventional thing is to use dummy variables to model these four conditions, but we won't do that because that makes it difficult to construct our priors. Instead, we'll build an index variable that encodes the different possibilities above.

```R
d$treatment <- 1 + d$prosoc_left + 2 * d$condition
```
```R
# verify using crosstabs (honestly not sure what this means)
xtabs(~ treatment + prosoc_left + condition, d)
```
    , , condition = 0
    
             prosoc_left
    treatment   0   1
            1 126   0
            2   0 126
            3   0   0
            4   0   0
    
    , , condition = 1
    
             prosoc_left
    treatment   0   1
            1   0   0
            2   0   0
            3 126   0
            4   0 126



The model implied by the research question is

$$
\begin{align*}
L_i &\sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) &= \alpha_{\text{ACTOR}[i]} + \beta_{\text{TREATMENT}[i]} \\
\alpha_j &\sim \text{to be determined} \\
\beta_j &\sim \text{to be determined} \\
\end{align*}
$$

Here $L_i$ indicates the 0/1 variable `pulled_left`. Since there is just the one outcome, we could also use $L_i \sim \text{Bernoulli}(p)$; it doesn't make a difference.

The model implies 7 different $\alpha$s (one for each of the chimpanzees) and 4 different $\beta$s, one for each of the four different treatment options. In principle we could also have each of the 7 chimps have their own $\beta$ value, but we'll do that in a later chapter; for now this is probably fine.

Now we need to determine that value of the priors. Let's consider a much simpler model

$$
\begin{align*}
L_i &\sim \text{Binomial(1, p)} \\
\text{logit}(p) &= \alpha \\
\alpha &\sim \text{Normal}(0, \omega)
\end{align*}
$$

As a very bad initla guess, let's start with something very flat like $\omega = 10$.


```R
m11.1 <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a,
        a ~ dnorm(0, 10)
    ),
    data=d
)
```

```R
set.seed(1999)
prior <- extract.prior(m11.1)
```

Now we need to convert the parameter to the outcome scale. This will require using the **logistic function** (inverse logit).

```R
p <- inv_logit(prior$a)
ggplot(data.frame(p=p), aes(p)) +
    geom_density(adjust=0.1)
```

    
![png](rethinking_ch11_output_12_0.png)
    


This is not what we expected (or wanted) - the model thinks that chimpanzees either never or always pull the lever! A flat outcome in logit space *is not* the same as a flat outcome in probability space. Let's try this again but with $\omega = 1.5$.


```R
m11.1_revised <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a,
        a ~ dnorm(0, 1.5)
    ),
    data=d
)
set.seed(1999)
prior_revised <- extract.prior(m11.1_revised)
p_revised <- inv_logit(prior_revised$a)
plot_df <- data.frame(p=c(p, p_revised), omega=rep(c(10, 1.5), each=length(p)))
ggplot(plot_df, aes(x = p, group = omega, colour = omega)) +
    geom_density(adjust=0.1)
```


    
![png](rethinking_ch11_output_14_0.png)
    


This seems far more plausible, with values in the middle being approximately flat but ones at the end being unlikely.

Now we need the priors for the $\beta$ s. Again, let's see what happens with conventional flat priors:


```R
m11.2 <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a + b[treatment],
        a ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 10)
    ),
    data=d
)

set.seed(1999)

prior <- extract.prior(m11.2, n=1e4)
p <- sapply(1:4, function(k) inv_logit(prior$a + prior$b[, k]))
```


```R
head(prior)
```


<dl>
	<dt>$a</dt>
		<dd><style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1.0990087329525</li><li>-0.0567445654723846</li><li>1.80451370523971</li><li>2.20470303824258</li><li>0.200535443459088</li><li>0.779740868698894</li><li>-0.824073038550978</li><li>-1.77841193391556</li><li>-1.72091962783602</li><li>1.74111299757237</li><li>-0.0163285358706231</li><li>0.81808346295915</li><li>-1.91985035591353</li><li>-0.163620525333154</li><li>-0.382678390116821</li><li>0.940464439984995</li><li>-0.244854507908104</li><li>1.0119884381048</li><li>2.30386925680895</li><li>-0.353455781341935</li><li>0.207227131051468</li><li>0.537907697522022</li><li>-0.588377359604187</li><li>-0.61225972845978</li><li>1.88336645919917</li><li>-2.18889636446024</li><li>-0.16535897427418</li><li>0.0364893106813865</li><li>-0.211385539375347</li><li>-0.770130193238007</li><li>-1.92721684408015</li><li>2.17366577693033</li><li>-2.67573534793796</li><li>2.55677514266025</li><li>0.0665395177587311</li><li>2.51563770057431</li><li>-0.438246502618581</li><li>-1.9348825375021</li><li>3.65659269579237</li><li>-0.566058965912184</li><li>-0.473349145770402</li><li>1.07400082976006</li><li>0.393835012010423</li><li>0.110333621858107</li><li>1.31084241693383</li><li>-0.775672634258939</li><li>0.193233185618424</li><li>0.240715500651279</li><li>-2.78201471637635</li><li>-0.909985561654701</li><li>2.64850871524962</li><li>-0.297357188087977</li><li>1.10603981497339</li><li>1.3104965185833</li><li>-0.377145712760375</li><li>0.022073908846177</li><li>-1.4008226370665</li><li>2.11750101307503</li><li>0.726581901004115</li><li>1.55752588284723</li><li>1.17480690476765</li><li>1.12078860733983</li><li>0.996433293155503</li><li>1.81731844934057</li><li>0.873086807023049</li><li>-0.12504891812232</li><li>-0.396346006066508</li><li>-1.23685082755814</li><li>0.242353365516451</li><li>0.972178153251967</li><li>-1.52592904015998</li><li>1.99302127673309</li><li>-2.936624284887</li><li>1.55709550809589</li><li>-0.072505191138886</li><li>-0.0504279518257073</li><li>-0.300786454698257</li><li>-1.1505612247834</li><li>-2.39571098045882</li><li>0.992065258947394</li><li>-1.30559974290832</li><li>0.94629835789906</li><li>2.28066477656889</li><li>-1.19825732177868</li><li>-0.495379458569176</li><li>2.789320623841</li><li>0.384275665951046</li><li>1.88645543720082</li><li>3.28086710035588</li><li>0.17843999789505</li><li>-2.03538063312531</li><li>0.299317264744412</li><li>0.235925313526689</li><li>0.49166990739309</li><li>-3.48515315080261</li><li>-0.404890680729558</li><li>-1.03428400190503</li><li>-0.240344959694099</li><li>-0.808763496419412</li><li>-0.615940638958182</li><li>0.0671462732300764</li><li>-0.245613382492046</li><li>-2.65303251793289</li><li>-1.19323669464306</li><li>0.239172542064238</li><li>-0.176273831082995</li><li>-0.912043331484706</li><li>-0.60614189291232</li><li>0.999181250794122</li><li>-0.199638267747965</li><li>-2.44218984212535</li><li>2.41015878581525</li><li>0.313813996531229</li><li>3.09553678917036</li><li>-0.0960977933768331</li><li>-1.93383675392094</li><li>0.280481251286326</li><li>1.2182601328488</li><li>-1.12310961555087</li><li>0.590401810796013</li><li>-1.16208232633618</li><li>-0.474987632545444</li><li>-0.489364754852076</li><li>-0.340165203472203</li><li>0.646628263841561</li><li>0.568969510490626</li><li>-0.356030396979487</li><li>0.441220058217153</li><li>0.450432877492011</li><li>0.0454152293542609</li><li>0.668126053897267</li><li>-2.37729814420032</li><li>-0.366468597145336</li><li>1.6925885051384</li><li>2.45198816273834</li><li>0.63543100858484</li><li>-1.22811921859081</li><li>0.739955541182074</li><li>1.15886024326776</li><li>1.13277844012657</li><li>1.62539261573929</li><li>-0.0743181193731782</li><li>1.26050967353945</li><li>-0.668764084826826</li><li>-1.31778400776038</li><li>0.514316310605948</li><li>0.219012913129383</li><li>-0.409330458774573</li><li>1.27937403906442</li><li>-0.0745910481386514</li><li>-1.76114612286284</li><li>0.793698034253496</li><li>0.912010566220838</li><li>-1.14933951254999</li><li>-0.57959489917956</li><li>2.83696256051142</li><li>-0.362586173301426</li><li>-1.26653976800457</li><li>0.756666529037035</li><li>-1.14620582391588</li><li>1.54612218282483</li><li>-1.63764132403711</li><li>2.41854056496868</li><li>1.53300389750871</li><li>-0.773043772832696</li><li>-0.552013565420262</li><li>1.02782642569344</li><li>-1.50655234080683</li><li>1.09978108708106</li><li>2.2006444523184</li><li>-0.667701448312748</li><li>-1.83570443087317</li><li>2.24794497784358</li><li>-1.38186422378822</li><li>0.46277357337418</li><li>1.41281107704495</li><li>3.10314829759639</li><li>-0.308284714089294</li><li>1.15201737217223</li><li>-0.819617632649657</li><li>-0.142760884085425</li><li>-1.182356708324</li><li>0.850992097587604</li><li>1.05914969468332</li><li>0.241254962222015</li><li>-1.55010246681313</li><li>-2.25069291986359</li><li>-0.276750190093097</li><li>1.91563159534673</li><li>0.575441369428926</li><li>2.09597750574915</li><li>1.98149073503606</li><li>1.27479302885238</li><li>2.45578002038848</li><li>1.00297487118845</li><li>-0.571832910212966</li><li>-1.1244779670948</li><li>-1.02531926370749</li><li>1.84101979525668</li><li>0.48810637195373</li><li>⋯</li><li>2.87501025817697</li><li>0.320784402563206</li><li>-1.2454679831571</li><li>3.11486040153197</li><li>-1.6004005576827</li><li>1.81154162496497</li><li>3.52146670447242</li><li>-4.12377127075576</li><li>0.0943705498654052</li><li>0.861254492223336</li><li>1.09288369009375</li><li>0.586756861387084</li><li>-0.194427633081143</li><li>-0.537972822038777</li><li>-1.58709738842695</li><li>0.508172207373916</li><li>-0.873787652457139</li><li>1.27146087567302</li><li>3.80353459074194</li><li>-0.78835417007325</li><li>-2.31098100377133</li><li>-0.900699794791351</li><li>-1.05994980766398</li><li>0.577896261265648</li><li>0.81134508429734</li><li>-0.594105777175059</li><li>-0.00441449272847745</li><li>-0.532687994756009</li><li>1.21901846234571</li><li>3.28580682349897</li><li>-2.76892292601527</li><li>-1.26709624521196</li><li>0.55054923778989</li><li>1.84068064180557</li><li>-2.55285961703919</li><li>1.369842668029</li><li>-0.400316047570987</li><li>-0.310370011594254</li><li>-1.63735558581869</li><li>-0.67695069654606</li><li>0.676312272668559</li><li>-2.14683014540286</li><li>-1.97619245713635</li><li>-1.45926273131771</li><li>-0.891646050334931</li><li>0.281080971697152</li><li>0.724269319170488</li><li>1.00006479443985</li><li>1.12262728419604</li><li>0.291281997877244</li><li>-2.09777500394879</li><li>1.59571101685692</li><li>1.6647893414784</li><li>-2.16783362912374</li><li>0.292671380760363</li><li>0.913514991992782</li><li>1.94883745449079</li><li>0.116466954120089</li><li>-1.0524695320208</li><li>1.71694437510792</li><li>1.90369454470825</li><li>-0.414080591135559</li><li>0.606033842767979</li><li>-0.98774001126712</li><li>2.28051243901412</li><li>-0.0945989171622975</li><li>2.0981259689675</li><li>0.469082997731544</li><li>-0.507320560646686</li><li>0.68618034617416</li><li>1.5343910016204</li><li>-0.550430806694815</li><li>-0.318887951679235</li><li>-0.0645463130380225</li><li>0.756041256781746</li><li>0.913417935726907</li><li>-0.0570342126462296</li><li>-0.0821434135089054</li><li>1.88615113004573</li><li>-0.774386591031534</li><li>0.672166648222155</li><li>-1.70556158356127</li><li>-1.68134417693016</li><li>0.260205769544765</li><li>-2.18571224091391</li><li>-1.50706419824241</li><li>1.29971788311207</li><li>-0.47411030853007</li><li>2.00384002270673</li><li>-1.17725208929731</li><li>1.17934499313447</li><li>-0.40827851774874</li><li>-0.994170680563371</li><li>1.0976519403026</li><li>1.06752766078969</li><li>2.28713965395861</li><li>-0.415088966196413</li><li>-0.433497886795327</li><li>-2.61006913400088</li><li>1.8880388498194</li><li>-1.52279307744807</li><li>1.48661808409958</li><li>-1.24541516363924</li><li>-1.92661520326133</li><li>-3.14072019569338</li><li>-0.11018790422208</li><li>-2.33144820507675</li><li>1.04587484166106</li><li>-0.00322723373323175</li><li>0.265969522089801</li><li>1.83281824400894</li><li>-1.61064336334012</li><li>-1.42076225684339</li><li>0.202794079555737</li><li>1.54619077441158</li><li>2.55706476551068</li><li>-0.072265333198774</li><li>1.17960314210263</li><li>-0.916940076324932</li><li>1.83514504854926</li><li>0.383649041040435</li><li>1.18251732017753</li><li>-0.0760268959802886</li><li>0.435679684188009</li><li>-0.218948924310393</li><li>-1.81599773349684</li><li>-0.34511112587323</li><li>1.49357761521626</li><li>-1.70194373017931</li><li>1.93588298692571</li><li>-2.07639300347615</li><li>2.42557513917244</li><li>-2.5952173887798</li><li>2.67337961718906</li><li>0.693013289837706</li><li>4.12359090593791</li><li>0.517571068444419</li><li>-0.165439626706809</li><li>-2.68642263929235</li><li>-0.548745606630313</li><li>2.79849988757242</li><li>1.33705671352023</li><li>-2.11217643550267</li><li>-0.535543969401976</li><li>0.636516455471811</li><li>1.62146896514032</li><li>1.43469302931568</li><li>1.99621100383905</li><li>0.25413277460457</li><li>-0.308869359788505</li><li>1.67407720051487</li><li>-0.715589007877677</li><li>-1.63463289717397</li><li>1.67187075559511</li><li>-0.78453286042005</li><li>-2.23717223313947</li><li>-1.8846584352114</li><li>0.166362572764357</li><li>-3.14384608236164</li><li>-0.598276043557425</li><li>-0.689260018093264</li><li>2.72407397498372</li><li>-0.32251118155836</li><li>-0.4984948081222</li><li>0.0551343319347003</li><li>-1.48987627736255</li><li>0.750584817940655</li><li>0.11851297462838</li><li>0.298666429666479</li><li>-0.0815783741904443</li><li>-2.56115342874834</li><li>-2.24852000300945</li><li>0.876442987069922</li><li>-0.297423062323906</li><li>0.498407436039938</li><li>-1.58457705443868</li><li>-1.2793604692827</li><li>1.24868265943584</li><li>0.665756213457642</li><li>-0.373110515322561</li><li>0.738926735993677</li><li>-0.644569739605611</li><li>-1.55878597271525</li><li>-1.68054612512767</li><li>0.0718207108644338</li><li>-2.2541875719008</li><li>-1.416041502296</li><li>0.174380519206576</li><li>0.47451020764916</li><li>1.87073868636485</li><li>-2.21757979701041</li><li>0.764276017085692</li><li>0.282843464181395</li><li>-1.93962954057248</li><li>-2.47066015498155</li><li>1.16569498348152</li><li>0.722883278583508</li><li>0.472988017503323</li><li>1.45271704561661</li><li>1.49930198060519</li></ol>
</dd>
	<dt>$b</dt>
		<dd><table class="dataframe">
<caption>A matrix: 10000 × 4 of type dbl</caption>
<tbody>
	<tr><td>  2.3442568</td><td>  6.3764744</td><td> -3.7080723</td><td>  1.9265919</td></tr>
	<tr><td>-11.4545612</td><td> 29.5329687</td><td>  3.4452271</td><td> 13.8832050</td></tr>
	<tr><td>  4.5335731</td><td>  7.5955124</td><td> -9.7822556</td><td> -6.8759615</td></tr>
	<tr><td> 14.0475584</td><td>-23.6633028</td><td>  8.3692817</td><td> 11.2265443</td></tr>
	<tr><td> -8.9711082</td><td>  2.2242951</td><td>  0.5369367</td><td>  7.1986007</td></tr>
	<tr><td> 12.2251913</td><td> -9.9842142</td><td> -1.5710891</td><td>  0.7431563</td></tr>
	<tr><td>  8.9680398</td><td> 12.4225478</td><td> 15.9026269</td><td> -2.5972655</td></tr>
	<tr><td>-12.4755349</td><td> -9.8796262</td><td>-11.0328934</td><td>  4.0039403</td></tr>
	<tr><td> -1.3568781</td><td>  5.6420655</td><td>  0.6823787</td><td> -0.1199685</td></tr>
	<tr><td> -5.6863034</td><td>-10.4045767</td><td> 18.4653460</td><td> -1.9130580</td></tr>
	<tr><td>  3.1355925</td><td> -1.1138801</td><td> -3.3811900</td><td>  6.7326112</td></tr>
	<tr><td> -0.5198668</td><td>-11.0705170</td><td> 10.9881967</td><td> -4.1300558</td></tr>
	<tr><td> -5.2860128</td><td>  1.2911142</td><td>  8.7750519</td><td> -2.4363744</td></tr>
	<tr><td> 16.9841331</td><td> -2.5606050</td><td> -0.7193846</td><td> -4.8778554</td></tr>
	<tr><td> -2.4521691</td><td> -3.2051686</td><td> -3.6361179</td><td> -1.2495291</td></tr>
	<tr><td>  1.7795417</td><td>  0.1782840</td><td> -5.9234847</td><td>  2.0890180</td></tr>
	<tr><td> -1.7547462</td><td> -0.9542162</td><td>  0.6227389</td><td>-15.3547092</td></tr>
	<tr><td> 13.3578888</td><td> -4.3153818</td><td> -1.6695330</td><td> -7.9081704</td></tr>
	<tr><td> 20.2885444</td><td>  6.0269986</td><td> 16.8712230</td><td>-15.8856106</td></tr>
	<tr><td> 19.1163312</td><td> 22.8583275</td><td> 13.5609907</td><td>-11.0182108</td></tr>
	<tr><td>  1.2259365</td><td> -3.3533088</td><td>  4.5009590</td><td>  0.1200125</td></tr>
	<tr><td> 24.0509662</td><td> -1.8680508</td><td>  7.7865223</td><td>  2.9445930</td></tr>
	<tr><td>-34.7499060</td><td>-16.2230816</td><td>  0.9881413</td><td>-22.6734565</td></tr>
	<tr><td> -2.2716444</td><td> -0.4441275</td><td> -4.6519103</td><td> -5.7334682</td></tr>
	<tr><td> -4.6375794</td><td>  1.6122342</td><td>  0.9616170</td><td> 20.3643188</td></tr>
	<tr><td> -7.8881934</td><td>  3.4170802</td><td>  2.8468086</td><td> -0.5764845</td></tr>
	<tr><td>  4.4219592</td><td>  3.6909453</td><td> 12.0476898</td><td>  3.0787480</td></tr>
	<tr><td> -9.3961772</td><td> 19.3353939</td><td>  9.9868094</td><td> -2.1090286</td></tr>
	<tr><td> 12.8575485</td><td>  3.3114821</td><td> 13.0189185</td><td> -2.2757185</td></tr>
	<tr><td>-14.2741644</td><td>  7.7307673</td><td>-17.4580176</td><td>  2.9178191</td></tr>
	<tr><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><td> -1.8439013</td><td>14.803115</td><td>  2.5540325</td><td> -3.7827610</td></tr>
	<tr><td>  6.9428391</td><td> 5.848275</td><td> -6.4156561</td><td> -1.3649017</td></tr>
	<tr><td> 11.4890670</td><td>-4.357761</td><td>  5.9417250</td><td>  8.0207503</td></tr>
	<tr><td> 15.9296570</td><td>-1.761103</td><td>  4.3414105</td><td> -2.5424384</td></tr>
	<tr><td>-12.9969561</td><td>-4.569553</td><td> -9.0190192</td><td>  0.1095436</td></tr>
	<tr><td>  0.5766694</td><td> 9.383386</td><td>  7.6315525</td><td> -0.8363032</td></tr>
	<tr><td>  1.9426887</td><td>-6.534219</td><td> 13.1785758</td><td>  7.8041578</td></tr>
	<tr><td> -6.3988240</td><td>-7.750173</td><td> 12.2871634</td><td> -7.0635414</td></tr>
	<tr><td> 13.8189124</td><td> 7.610849</td><td>  4.7482271</td><td> -4.5695114</td></tr>
	<tr><td> -6.9590950</td><td> 4.607531</td><td> 13.2680226</td><td> 18.0089205</td></tr>
	<tr><td> -1.1539500</td><td>-8.197833</td><td> 13.9161061</td><td> -5.6348440</td></tr>
	<tr><td> -3.6740682</td><td> 9.269816</td><td>-24.9949334</td><td> -5.2652554</td></tr>
	<tr><td> -1.3654391</td><td>-4.896839</td><td> -3.2749290</td><td>  8.7422905</td></tr>
	<tr><td>  1.9576198</td><td> 5.751136</td><td> 12.1871917</td><td>  0.7332203</td></tr>
	<tr><td>  1.1651788</td><td>-7.087735</td><td> -1.8780152</td><td>-11.9602392</td></tr>
	<tr><td> 16.8451647</td><td>-3.733043</td><td>  4.4231282</td><td> -3.8719501</td></tr>
	<tr><td> -4.3601785</td><td>-7.661046</td><td> -4.6358784</td><td> -9.7928787</td></tr>
	<tr><td>-23.3854914</td><td>19.168330</td><td>  2.6438037</td><td> -0.7126586</td></tr>
	<tr><td> 13.8016811</td><td>-8.247701</td><td>  4.0322728</td><td> -9.4303253</td></tr>
	<tr><td>  3.1382115</td><td> 3.862102</td><td> -6.9278462</td><td> -1.7582840</td></tr>
	<tr><td> 15.8122350</td><td> 5.478138</td><td>  7.2860846</td><td>-11.2076342</td></tr>
	<tr><td>-12.1356020</td><td>12.732871</td><td>  4.7656680</td><td> -1.6414040</td></tr>
	<tr><td> -1.3605389</td><td>-1.991165</td><td> 16.1243994</td><td>  7.4137730</td></tr>
	<tr><td>  4.0032980</td><td>-1.406819</td><td> -4.0184819</td><td>  9.4294299</td></tr>
	<tr><td>-31.7109551</td><td>-2.641941</td><td> 18.1742139</td><td>  4.0451019</td></tr>
	<tr><td>  0.4710924</td><td>-2.507952</td><td>  6.1259079</td><td>  5.6442585</td></tr>
	<tr><td> -3.8074373</td><td>-2.369074</td><td>  4.7317961</td><td>-11.2946394</td></tr>
	<tr><td> -3.0042858</td><td> 5.351514</td><td>  0.8566353</td><td> -4.3092898</td></tr>
	<tr><td> -6.9128245</td><td> 8.710579</td><td>  8.9491867</td><td> 10.6684413</td></tr>
	<tr><td>-22.2747536</td><td>-3.736537</td><td>  5.4091542</td><td>-15.8228765</td></tr>
</tbody>
</table>
</dd>
</dl>



This gives us the prior probability of the probability of pulling the left lever, but actually what we're interested in is the *difference* among the treatments.


```R
head(p)
```


<table class="dataframe">
<caption>A matrix: 6 × 4 of type dbl</caption>
<tbody>
	<tr><td>9.690297e-01</td><td>9.994335e-01</td><td>0.0685573765</td><td>0.953717371</td></tr>
	<tr><td>1.001611e-05</td><td>1.000000e+00</td><td>0.9673426413</td><td>0.999999011</td></tr>
	<tr><td>9.982354e-01</td><td>9.999173e-01</td><td>0.0003428955</td><td>0.006234222</td></tr>
	<tr><td>9.999999e-01</td><td>4.793453e-10</td><td>0.9999744279</td><td>0.999998531</td></tr>
	<tr><td>1.552106e-04</td><td>9.187013e-01</td><td>0.6764428282</td><td>0.999388593</td></tr>
	<tr><td>9.999978e-01</td><td>1.005783e-04</td><td>0.3118792593</td><td>0.820964713</td></tr>
</tbody>
</table>




```R
# here we are interested in the absolute difference for some reason?
difference_plot_df <- data.frame(
    diff = abs(p[, 1] - p[, 2]),
    omega = 10
)

ggplot(difference_plot_df, aes(diff, group = omega, colour = omega)) +
    geom_density(adjust=0.1)
```


    
![png](rethinking_ch11_output_20_0.png)
    


Again, this seems unlikely. We want more skeptical priors. Let's see what happens if we use a $\text{Normal}(0, 0.5)$ prior instead.


```R
m11.2_revised <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a + b[treatment],
        a ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data=d
)

set.seed(1999)

prior_revised <- extract.prior(m11.2_revised, n=1e4)
p_revised <- sapply(1:4, function(k) inv_logit(prior_revised$a + prior_revised$b[, k]))

difference_plot_df <- rbind(difference_plot_df, data.frame(
    diff = abs(p_revised[, 1] - p_revised[, 2]),
    omega = 0.5
))

ggplot(difference_plot_df, aes(diff, group = omega, colour = omega)) +
    geom_density(adjust=0.1)
```


    
![png](rethinking_ch11_output_22_0.png)
    


Now the probability is concentrated on low effect sizes, which is what we expect. We can also compute the mean effect size.


```R
m11.3 <- quap(
    alist(
        pulled_left <- dbinom(1, p),
        logit(p) <- a + b[treatment],
        a ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = d
)

set.seed(1999)

prior <- extract.prior(m11.3, n=1e4)
p <- sapply(1:4, function(k) inv_logit(prior$a + prior$b[, k]))
mean(abs(p[, 1] - p[, 2]))
```


0.0983866320386242


So the difference is roughly 10% (before we see the data).

Now lets analyze the model using **HMC**. The `quap` function will actually do OK, but only because of the regularizing priors.


```R
# trimmed data list
dat_list <- list(
    pulled_left = d$pulled_left,
    actor = d$actor,
    treatment = as.integer(d$treatment)
)
```


```R
m11.4 <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + b[treatment],
        a[actor] ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = dat_list,
    chains = 4,
    log_lik = TRUE
)

precis(m11.4, depth = 2)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.5 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.5 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.5 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.5 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.5 seconds.
    Total execution time: 2.3 seconds.
    



<table class="dataframe">
<caption>A precis: 11 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td>-0.44428874</td><td>0.3273410</td><td>-0.97968236</td><td> 0.06898334</td><td>1.003193</td><td> 698.7231</td></tr>
	<tr><th scope=row>a[2]</th><td> 3.91111090</td><td>0.7448156</td><td> 2.78077110</td><td> 5.18344610</td><td>1.002924</td><td>1418.6434</td></tr>
	<tr><th scope=row>a[3]</th><td>-0.75459323</td><td>0.3290882</td><td>-1.29314530</td><td>-0.23659085</td><td>1.002498</td><td> 715.8554</td></tr>
	<tr><th scope=row>a[4]</th><td>-0.75528052</td><td>0.3292270</td><td>-1.29220150</td><td>-0.23818057</td><td>1.006137</td><td> 701.6992</td></tr>
	<tr><th scope=row>a[5]</th><td>-0.45348305</td><td>0.3246345</td><td>-0.96770977</td><td> 0.08359580</td><td>1.003737</td><td> 774.5869</td></tr>
	<tr><th scope=row>a[6]</th><td> 0.46699897</td><td>0.3351792</td><td>-0.06458849</td><td> 0.99162481</td><td>1.005012</td><td> 792.3203</td></tr>
	<tr><th scope=row>a[7]</th><td> 1.95783006</td><td>0.4138242</td><td> 1.28397515</td><td> 2.64307420</td><td>1.002946</td><td> 917.9482</td></tr>
	<tr><th scope=row>b[1]</th><td>-0.03779445</td><td>0.2851385</td><td>-0.49886337</td><td> 0.43130246</td><td>1.002709</td><td> 675.4844</td></tr>
	<tr><th scope=row>b[2]</th><td> 0.48622780</td><td>0.2909519</td><td> 0.02550242</td><td> 0.96139139</td><td>1.003433</td><td> 685.0708</td></tr>
	<tr><th scope=row>b[3]</th><td>-0.38302442</td><td>0.2900145</td><td>-0.84469238</td><td> 0.08773202</td><td>1.003173</td><td> 698.6702</td></tr>
	<tr><th scope=row>b[4]</th><td> 0.37542829</td><td>0.2860251</td><td>-0.08431510</td><td> 0.82723556</td><td>1.003776</td><td> 713.4152</td></tr>
</tbody>
</table>




```R
par(bg = 'white')
post <- extract.samples(m11.4)
p_left <- inv_logit(post$a)
plot(precis(as.data.frame(p_left)), xlim=c(0, 1))
```


    
![png](rethinking_ch11_output_28_0.png)
    


These are the values for the individual chimpanzees. Note that four of the chimps have a preference for the left lever, and the rest have the opposite preference. Chimp 2 only ever pulled the left lever. This is a crazy amount of variation in individual preferences, but apparently quite normal in behavioural experiments.

Now let's look at the sidedness preferences


```R
par(bg = 'white')
labs <- c("R/N", "L/N", "R/P", "L/P")
plot(precis(m11.4, depth=2, pars="b"), labels=labs)
```


    
![png](rethinking_ch11_output_30_0.png)
    


In order to interpret this, we should consider what we expect. We are hoping to see more of the prosocial behaviour when there is a partner present, so we should compare the first / third rows and the second / fourth rows. Just from eyeballing it, it doesn't look like there is much of a difference. Let's compare the differences directly.


```R
par(bg = 'white')
diffs <- list(
    db13 = post$b[, 1] - post$b[, 3],
    db24 = post$b[, 2] - post$b[, 4]
)
plot(precis(diffs))
```


    
![png](rethinking_ch11_output_32_0.png)
    


The first one is comparing the behaviour when the prosocial option was on the right. In that case, we expect to see a *positive* difference as the chimp pulls the lever on the right more. The second one is comparing when the prosocial option was on the left. In that case, we expect to see a *negative* difference. From the above, there is some evidence for the right, but I am unconvinced when the option is on the left.

Now let's do a posterior predictive check. We'll summarize the proportions of left pulls for each actor in each treatment and then plot against the posterior predictions.


```R
pl <- by(d$pulled_left, list(d$actor, d$treatment), mean)
pl
```


    : 1
    : 1
    [1] 0.3333333
    ------------------------------------------------------------ 
    : 2
    : 1
    [1] 1
    ------------------------------------------------------------ 
    : 3
    : 1
    [1] 0.2777778
    ------------------------------------------------------------ 
    : 4
    : 1
    [1] 0.3333333
    ------------------------------------------------------------ 
    : 5
    : 1
    [1] 0.3333333
    ------------------------------------------------------------ 
    : 6
    : 1
    [1] 0.7777778
    ------------------------------------------------------------ 
    : 7
    : 1
    [1] 0.7777778
    ------------------------------------------------------------ 
    : 1
    : 2
    [1] 0.5
    ------------------------------------------------------------ 
    : 2
    : 2
    [1] 1
    ------------------------------------------------------------ 
    : 3
    : 2
    [1] 0.6111111
    ------------------------------------------------------------ 
    : 4
    : 2
    [1] 0.5
    ------------------------------------------------------------ 
    : 5
    : 2
    [1] 0.5555556
    ------------------------------------------------------------ 
    : 6
    : 2
    [1] 0.6111111
    ------------------------------------------------------------ 
    : 7
    : 2
    [1] 0.8333333
    ------------------------------------------------------------ 
    : 1
    : 3
    [1] 0.2777778
    ------------------------------------------------------------ 
    : 2
    : 3
    [1] 1
    ------------------------------------------------------------ 
    : 3
    : 3
    [1] 0.1666667
    ------------------------------------------------------------ 
    : 4
    : 3
    [1] 0.1111111
    ------------------------------------------------------------ 
    : 5
    : 3
    [1] 0.2777778
    ------------------------------------------------------------ 
    : 6
    : 3
    [1] 0.5555556
    ------------------------------------------------------------ 
    : 7
    : 3
    [1] 0.9444444
    ------------------------------------------------------------ 
    : 1
    : 4
    [1] 0.5555556
    ------------------------------------------------------------ 
    : 2
    : 4
    [1] 1
    ------------------------------------------------------------ 
    : 3
    : 4
    [1] 0.3333333
    ------------------------------------------------------------ 
    : 4
    : 4
    [1] 0.4444444
    ------------------------------------------------------------ 
    : 5
    : 4
    [1] 0.5
    ------------------------------------------------------------ 
    : 6
    : 4
    [1] 0.6111111
    ------------------------------------------------------------ 
    : 7
    : 4
    [1] 1


`pl` is a matrix with 7 rows and 4 columns. Each row is a chimp, and each column is one of the treatment conditions. Each cell is the proportion of left pulls for that chimp and the specified treatment condition.


```R
par(bg = 'white')
plot(NULL, xlim=c(1, 28), ylim=c(0, 1), xlab="", ylab="Proportion Left Lever", xaxt="n", yaxt="n")
axis(2, at=c(0, 0.5, 1), labels=c(0, 0.5, 1))
abline(h=0.5, lty=2)
for (j in 1:7) abline(v=(j-1) * 4 + 4.5, lwd=0.5) # draw the vertical lines
for (j in 1:7) text((j-1) * 4 + 2.5, 1.1, concat("Actor", j), xpd=TRUE) # label each grid segment
for (j in 1:7[-2]) {
    lines( (j-1)*4 + c(1, 3), pl[j, c(1, 3)], lwd=2, col=rangi2) # prosocial on the right
    lines( (j-1)*4 + c(2, 4), pl[j, c(2, 4)], lwd=2, col=rangi2) # prosocial on the left
}

# adding in the filled and hollow points
points(1:28, t(pl), pch=16, col="white", cex=1.7) 
points(1:28, t(pl), pch=c(1, 1, 16, 16), col=rangi2, lwd=2)

yoff <- 0.01
text(1, pl[1,1] - yoff, "R/N", pos=1, cex=0.8)
text(2, pl[1,2] - yoff, "L/N", pos=3, cex=0.8)
text(3, pl[1,3] - yoff, "R/P", pos=1, cex=0.8)
text(4, pl[1,4] - yoff, "L/P", pos=3, cex=0.8)
mtext("Observed proportion")
```


    
![png](rethinking_ch11_output_36_0.png)
    



```R
# getting the posterior predictions
dat <- list(actor=rep(1:7, each=4), treatment=rep(1:4, times=7))
p_post <- link(m11.4, data=dat)
p_mu <- apply(p_post, 2, mean)
p_ci <- apply(p_post, 2, PI)
```

In theory I could plot this, but...

Basically, the model shows almost no expected change when adding a partner. Most of the variation seems to come from the actors themselves, rather than from the presence or absence of a partner.

Interestingly, it seems that the chimps have different levels of susceptibility to the presence / absence of a partner. It might be interesting to allow each chimp to have their own parameters, but we'll do that when we talk about mutilevel models.

We haven't considered a model that splits into separate index variables the location of the prosocial option and the presence of a partner. This is because the hypothesis that we're working under is that there is an interaction effect going on - the prosocial choice will happen more often *when there is a partner present*. We could also build a model without the interaction and use WAIC or PSIS to compare them.

From the above, we can guess that the simpler model (no interaction) will do just fine; there doesn't seem to be much of an effect anyway.


```R
d$side <- d$prosoc_left + 1 # right 1, left 2
d$cond <- d$condition + 1 # no partner 1, partner 2

# the model. We add the log_lik so we can compare the models
data_list2 <- list(
    pulled_left = d$pulled_left,
    actor = d$actor,
    side = d$side,
    cond = d$cond
)

m11.5 <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + bs[side] + bc[cond],
        a[actor] ~ dnorm(0, 1.5),
        bs[side] ~ dnorm(0, 0.5),
        bc[cond] ~ dnorm(0, 0.5)
    ),
    data=data_list2,
    chains=4,
    log_lik=TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.7 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.7 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.7 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.7 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.7 seconds.
    Total execution time: 3.0 seconds.
    



```R
compare(m11.5, m11.4, func = PSIS)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>PSIS</th><th scope=col>SE</th><th scope=col>dPSIS</th><th scope=col>dSE</th><th scope=col>pPSIS</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m11.5</th><td>530.4549</td><td>19.10183</td><td>0.000000</td><td>      NA</td><td>7.626884</td><td>0.6220271</td></tr>
	<tr><th scope=row>m11.4</th><td>531.4513</td><td>18.94224</td><td>0.996322</td><td>1.279907</td><td>8.152326</td><td>0.3779729</td></tr>
</tbody>
</table>



The PSIS scores are basically identical and well within the standard error.

### 1.2 Relative shark and absolute deer

Earlier we were focused on the absolute difference in the outcome scale - how much does the probability change? With logistic regressions it is common to instead compare the *relative effects* - that is, instead of looking at the absolute change in the probability, focus on the *relative change* - i.e. did the odds double?

We can calculate the proportional odds by exponentiating the parameters.


```R
# relative odds of switching from treatment 2 to treatment 4 (adding a partner)
post <- extract.samples(m11.4)
mean(exp(post$b[, 4] - post$b[, 2]))
```


0.929158500832028


This tells us that on average, adding a partner multiplies the odds by 92% (a reduction of 8%).

The risk of using relative odds is that it isn't enough to tell whether the variable is important or not. If the other parameters make the outcome unlikely, then even a very large change in the proportional odds doesn't mean much.

Of course, sometimes the relative risk is very important (for instance, when everything else is being held constant). Analogy: absolute risk of dying by deer is much higher than being attacked by a shark. However, when you're swimming in the ocean you care about the relative risk of shark attacks; conditional on being in the ocean, sharks are much more dangerous than deer.

### 1.3 Aggregated binomial: Chimpanzees again, condensed

In the chimpanzee data, the models all calculated the likelihood of observing either 0 or 1 pulls; this was convenient because that's how the data were organized. We could just as easily have calculated the total count of left pulls for each condition.


```R
data(chimpanzees)
d <- chimpanzees
d$treatment <- 1 + d$prosoc_left + 2 * d$condition
d$side <- d$prosoc_left + 1 # right 1, left 2
d$cond <- d$condition + 1 # no partner 1, partner 2

d_aggregated <- aggregate(
    d$pulled_left,
    list(
        treatment = d$treatment,
        actor = d$actor,
        side = d$side,
        cond = d$cond
    ),
    sum
)
colnames(d_aggregated)[5] <- "left_pulls"
head(d_aggregated)
```


<table class="dataframe">
<caption>A data.frame: 6 × 5</caption>
<thead>
	<tr><th></th><th scope=col>treatment</th><th scope=col>actor</th><th scope=col>side</th><th scope=col>cond</th><th scope=col>left_pulls</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>1</td><td>1</td><td>1</td><td> 6</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>2</td><td>1</td><td>1</td><td>18</td></tr>
	<tr><th scope=row>3</th><td>1</td><td>3</td><td>1</td><td>1</td><td> 5</td></tr>
	<tr><th scope=row>4</th><td>1</td><td>4</td><td>1</td><td>1</td><td> 6</td></tr>
	<tr><th scope=row>5</th><td>1</td><td>5</td><td>1</td><td>1</td><td> 6</td></tr>
	<tr><th scope=row>6</th><td>1</td><td>6</td><td>1</td><td>1</td><td>14</td></tr>
</tbody>
</table>



Now we can get the same kind of model, but using the aggregated counts.


```R
dat <- with(d_aggregated, list(
    left_pulls = left_pulls,
    treatment = treatment,
    actor = actor,
    side = side,
    cond = cond
))

m11.6 <- ulam(
    alist(
        left_pulls ~ dbinom(18, p),
        logit(p) <- a[actor] + b[treatment],
        a[actor] ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.1 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.
    



```R
par(bg = 'white')
plot(precis(m11.6, depth = 2))
```


    
![png](rethinking_ch11_output_49_0.png)
    



```R
compare(m11.6, m11.4, func=PSIS)
```

    Warning message in compare(m11.6, m11.4, func = PSIS):
    “Different numbers of observations found for at least two models.
    Model comparison is valid only for models fit to exactly the same observations.
    Number of observations for each model:
    m11.6 28 
    m11.4 504 
    ”
    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    



<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>PSIS</th><th scope=col>SE</th><th scope=col>dPSIS</th><th scope=col>dSE</th><th scope=col>pPSIS</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m11.6</th><td>114.2045</td><td> 8.529007</td><td>  0.0000</td><td>      NA</td><td>8.417584</td><td>1.000000e+00</td></tr>
	<tr><th scope=row>m11.4</th><td>531.4513</td><td>18.942245</td><td>417.2468</td><td>9.531782</td><td>8.152326</td><td>2.488911e-91</td></tr>
</tbody>
</table>



First issue is that the PSIS scores are different. This is purely a result of how we organized the data (the aggregation process). If we see 6 successes in 9 trials, the probability of that happening is 

$$
\frac{6!}{9!(9-6)!}p^6 (1-p)^{9-6}
$$

whereas if we take the actual six successes we got, that would not have the term out front

$$
p^6 (1-p)^{9-6}
$$

basically, the extra uncertainty from the fact that we are throwing away information about the order is causing that.

Also, since we trained the models on different data, we can't really compare them anyway.

The warning about the Pareto-$k$ value is due to the fact that since we are aggregating the data, each time we leave out one piece of the data we are really leaving out 18 pieces of data - all of the trials for one of the chimpanzees. This makes the remaining points more influential.

Bottom line: if you want to compare models using PSIS, you shouldn't aggregate the counts.

### 1.4 Aggregated binomial: Graduate-school admissions

In the above example, all of the chimpanzees had the same number of trials. That is often not the case. Let's see how to deal with that by taking a look at some graduate school admissions data. These are the graduate school applications for UC Berkley graduate programs.


```R
data(UCBadmit)
d <- UCBadmit
d
```


<table class="dataframe">
<caption>A data.frame: 12 × 5</caption>
<thead>
	<tr><th></th><th scope=col>dept</th><th scope=col>applicant.gender</th><th scope=col>admit</th><th scope=col>reject</th><th scope=col>applications</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>A</td><td>male  </td><td>512</td><td>313</td><td>825</td></tr>
	<tr><th scope=row>2</th><td>A</td><td>female</td><td> 89</td><td> 19</td><td>108</td></tr>
	<tr><th scope=row>3</th><td>B</td><td>male  </td><td>353</td><td>207</td><td>560</td></tr>
	<tr><th scope=row>4</th><td>B</td><td>female</td><td> 17</td><td>  8</td><td> 25</td></tr>
	<tr><th scope=row>5</th><td>C</td><td>male  </td><td>120</td><td>205</td><td>325</td></tr>
	<tr><th scope=row>6</th><td>C</td><td>female</td><td>202</td><td>391</td><td>593</td></tr>
	<tr><th scope=row>7</th><td>D</td><td>male  </td><td>138</td><td>279</td><td>417</td></tr>
	<tr><th scope=row>8</th><td>D</td><td>female</td><td>131</td><td>244</td><td>375</td></tr>
	<tr><th scope=row>9</th><td>E</td><td>male  </td><td> 53</td><td>138</td><td>191</td></tr>
	<tr><th scope=row>10</th><td>E</td><td>female</td><td> 94</td><td>299</td><td>393</td></tr>
	<tr><th scope=row>11</th><td>F</td><td>male  </td><td> 22</td><td>351</td><td>373</td></tr>
	<tr><th scope=row>12</th><td>F</td><td>female</td><td> 24</td><td>317</td><td>341</td></tr>
</tbody>
</table>



We're going to look to see if we can find a gender bias here; that is, model probability of admittance as a function of gender.

$$
\begin{align*}
A_i &\sim \text{Binomial}(N_i, p) \\
\text{logit}(p) &= \alpha_{ \text{GID}[i] } \\
\alpha_j &\sim \text{Normal}(0, 1.5)
\end{align*}
$$

Here, $N_i$ is the number of applicants to that program / department and GID is the indexed gender (1 -> male, 2 -> female).


```R
dat_list <- list(
    admit = d$admit,
    applications = d$applications,
    gid = ifelse(d$applicant.gender == "male", 1, 2)
)

m11.7 <- ulam(
    alist(
        admit ~ dbinom(applications, p),
        logit(p) <- a[gid],
        a[gid] ~ dnorm(0, 1.5)
    ),
    data = dat_list,
    chains = 4
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.
    



```R
par(bg = 'white')
precis(m11.7, depth = 2)
plot(precis(m11.7, depth = 2))
```


<table class="dataframe">
<caption>A precis: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td>-0.2215393</td><td>0.03767883</td><td>-0.2809970</td><td>-0.1655059</td><td>0.9999707</td><td>1506.931</td></tr>
	<tr><th scope=row>a[2]</th><td>-0.8297312</td><td>0.05121146</td><td>-0.9115474</td><td>-0.7478981</td><td>1.0031896</td><td>1396.420</td></tr>
</tbody>
</table>




    
![png](rethinking_ch11_output_55_1.png)
    


So the mean for males is higher than for females (on the logit scale). Let's see how much higher relatively (shark) and absolutely (relative deer).


```R
post <- extract.samples(m11.7)
diff_a <- post$a[, 1] - post$a[, 2]
diff_p <- inv_logit(post$a[, 1]) - inv_logit(post$a[, 2])
precis(list(diff_a=diff_a, diff_p=diff_p))
```


<table class="dataframe">
<caption>A precis: 2 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>diff_a</th><td>0.6081919</td><td>0.06430635</td><td>0.5035140</td><td>0.7125553</td><td>▁▁▁▁▃▇▇▃▂▁▁ </td></tr>
	<tr><th scope=row>diff_p</th><td>0.1410493</td><td>0.01443811</td><td>0.1175834</td><td>0.1645494</td><td>▁▁▁▁▂▃▇▇▃▂▁▁</td></tr>
</tbody>
</table>



So the probability for males is definitely higher - about 12% - 16% higher.

Let's speculate as to the cause of this. Plot the posterior predictions.


```R
par(bg = 'white')
postcheck(m11.7)

# connect the points from the same department
for (i in 1:6) {
    x <- 1 + 2 * (i - 1)
    y1 <- d$admit[x] / d$applications[x]
    y2 <- d$admit[x + 1] / d$applications[x + 1]
    lines(c(x, x+1), c(y1, y2), col = rangi2, lwd = 2)
    text(x + 0.5, (y1 + y2) / 2 + 0.05, d$dept[x], cex = 0.8, col = rangi2)
}
```


    
![png](rethinking_ch11_output_59_0.png)
    


These predictions are pretty awful; only two departments had a lower rate of admission than men, yet the model predicts that the rate for women should be 14% lower.

This is an example of Simpson's Paradox.

Basically, the model is answering the question that we gave it - across *all departments*, what is the difference in admission rates for men and women? But women did not apply equally to all departments, and we are seeing the effect of this. In general, women tended not to apply to departments (A & B) with high aceptance rates and instead applied to ones with low acceptance rates (E & F).

We probably want to ask "What is the average difference in probability of admission between women and men within the same department?". We need to estimate the admission rate within the different departments.

$$
\begin{align*}
A_i &\sim \text{Binomial}(N_i, p_i) \\
\text{logi}(p) &= \alpha_{\text{GID}[i]} + \delta_{\text{DEPT}[i]} \\
\alpha_j &\sim \text{Normal}(0, 1.5) \\
\delta_k &\sim \text{Normal}(0, 1.5)
\end{align*}
$$

So this model has a universal male / female adjustment ($\alpha$) along with a department-specific adjustment ($\delta$).


```R
# adding in the departmental index
dat_list$dept_id <- rep(1:6, each = 2)

m11.8 <- ulam(
    alist(
        admit ~ dbinom(applications, p),
        logit(p) <- a[gid] + delta[dept_id],
        a[gid] ~ dnorm(0, 1.5),
        delta[dept_id] ~ dnorm(0, 1.5)
    ),
    data = dat_list,
    chains = 4,
    iter = 4000
)

```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
    Chain 1 Iteration:    1 / 4000 [  0%]  (Warmup) 
    Chain 1 Iteration:  100 / 4000 [  2%]  (Warmup) 
    Chain 1 Iteration:  200 / 4000 [  5%]  (Warmup) 
    Chain 1 Iteration:  300 / 4000 [  7%]  (Warmup) 
    Chain 1 Iteration:  400 / 4000 [ 10%]  (Warmup) 
    Chain 1 Iteration:  500 / 4000 [ 12%]  (Warmup) 
    Chain 1 Iteration:  600 / 4000 [ 15%]  (Warmup) 
    Chain 1 Iteration:  700 / 4000 [ 17%]  (Warmup) 
    Chain 1 Iteration:  800 / 4000 [ 20%]  (Warmup) 
    Chain 1 Iteration:  900 / 4000 [ 22%]  (Warmup) 
    Chain 1 Iteration: 1000 / 4000 [ 25%]  (Warmup) 
    Chain 1 Iteration: 1100 / 4000 [ 27%]  (Warmup) 
    Chain 1 Iteration: 1200 / 4000 [ 30%]  (Warmup) 
    Chain 1 Iteration: 1300 / 4000 [ 32%]  (Warmup) 
    Chain 1 Iteration: 1400 / 4000 [ 35%]  (Warmup) 
    Chain 1 Iteration: 1500 / 4000 [ 37%]  (Warmup) 
    Chain 1 Iteration: 1600 / 4000 [ 40%]  (Warmup) 
    Chain 1 Iteration: 1700 / 4000 [ 42%]  (Warmup) 
    Chain 1 Iteration: 1800 / 4000 [ 45%]  (Warmup) 
    Chain 1 Iteration: 1900 / 4000 [ 47%]  (Warmup) 
    Chain 1 Iteration: 2000 / 4000 [ 50%]  (Warmup) 
    Chain 1 Iteration: 2001 / 4000 [ 50%]  (Sampling) 
    Chain 1 Iteration: 2100 / 4000 [ 52%]  (Sampling) 
    Chain 1 Iteration: 2200 / 4000 [ 55%]  (Sampling) 
    Chain 1 Iteration: 2300 / 4000 [ 57%]  (Sampling) 
    Chain 1 Iteration: 2400 / 4000 [ 60%]  (Sampling) 
    Chain 1 Iteration: 2500 / 4000 [ 62%]  (Sampling) 
    Chain 1 Iteration: 2600 / 4000 [ 65%]  (Sampling) 
    Chain 1 Iteration: 2700 / 4000 [ 67%]  (Sampling) 
    Chain 1 Iteration: 2800 / 4000 [ 70%]  (Sampling) 
    Chain 1 Iteration: 2900 / 4000 [ 72%]  (Sampling) 
    Chain 1 Iteration: 3000 / 4000 [ 75%]  (Sampling) 
    Chain 1 Iteration: 3100 / 4000 [ 77%]  (Sampling) 
    Chain 1 Iteration: 3200 / 4000 [ 80%]  (Sampling) 
    Chain 1 Iteration: 3300 / 4000 [ 82%]  (Sampling) 
    Chain 1 Iteration: 3400 / 4000 [ 85%]  (Sampling) 
    Chain 1 Iteration: 3500 / 4000 [ 87%]  (Sampling) 
    Chain 1 Iteration: 3600 / 4000 [ 90%]  (Sampling) 
    Chain 1 Iteration: 3700 / 4000 [ 92%]  (Sampling) 
    Chain 1 Iteration: 3800 / 4000 [ 95%]  (Sampling) 
    Chain 1 Iteration: 3900 / 4000 [ 97%]  (Sampling) 
    Chain 1 Iteration: 4000 / 4000 [100%]  (Sampling) 
    Chain 1 finished in 0.3 seconds.
    Chain 2 Iteration:    1 / 4000 [  0%]  (Warmup) 
    Chain 2 Iteration:  100 / 4000 [  2%]  (Warmup) 
    Chain 2 Iteration:  200 / 4000 [  5%]  (Warmup) 
    Chain 2 Iteration:  300 / 4000 [  7%]  (Warmup) 
    Chain 2 Iteration:  400 / 4000 [ 10%]  (Warmup) 
    Chain 2 Iteration:  500 / 4000 [ 12%]  (Warmup) 
    Chain 2 Iteration:  600 / 4000 [ 15%]  (Warmup) 
    Chain 2 Iteration:  700 / 4000 [ 17%]  (Warmup) 
    Chain 2 Iteration:  800 / 4000 [ 20%]  (Warmup) 
    Chain 2 Iteration:  900 / 4000 [ 22%]  (Warmup) 
    Chain 2 Iteration: 1000 / 4000 [ 25%]  (Warmup) 
    Chain 2 Iteration: 1100 / 4000 [ 27%]  (Warmup) 
    Chain 2 Iteration: 1200 / 4000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 1300 / 4000 [ 32%]  (Warmup) 
    Chain 2 Iteration: 1400 / 4000 [ 35%]  (Warmup) 
    Chain 2 Iteration: 1500 / 4000 [ 37%]  (Warmup) 
    Chain 2 Iteration: 1600 / 4000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 1700 / 4000 [ 42%]  (Warmup) 
    Chain 2 Iteration: 1800 / 4000 [ 45%]  (Warmup) 
    Chain 2 Iteration: 1900 / 4000 [ 47%]  (Warmup) 
    Chain 2 Iteration: 2000 / 4000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 2001 / 4000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 2100 / 4000 [ 52%]  (Sampling) 
    Chain 2 Iteration: 2200 / 4000 [ 55%]  (Sampling) 
    Chain 2 Iteration: 2300 / 4000 [ 57%]  (Sampling) 
    Chain 2 Iteration: 2400 / 4000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 2500 / 4000 [ 62%]  (Sampling) 
    Chain 2 Iteration: 2600 / 4000 [ 65%]  (Sampling) 
    Chain 2 Iteration: 2700 / 4000 [ 67%]  (Sampling) 
    Chain 2 Iteration: 2800 / 4000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 2900 / 4000 [ 72%]  (Sampling) 
    Chain 2 Iteration: 3000 / 4000 [ 75%]  (Sampling) 
    Chain 2 Iteration: 3100 / 4000 [ 77%]  (Sampling) 
    Chain 2 Iteration: 3200 / 4000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 3300 / 4000 [ 82%]  (Sampling) 
    Chain 2 Iteration: 3400 / 4000 [ 85%]  (Sampling) 
    Chain 2 Iteration: 3500 / 4000 [ 87%]  (Sampling) 
    Chain 2 Iteration: 3600 / 4000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 3700 / 4000 [ 92%]  (Sampling) 
    Chain 2 Iteration: 3800 / 4000 [ 95%]  (Sampling) 
    Chain 2 Iteration: 3900 / 4000 [ 97%]  (Sampling) 
    Chain 2 Iteration: 4000 / 4000 [100%]  (Sampling) 
    Chain 2 finished in 0.3 seconds.
    Chain 3 Iteration:    1 / 4000 [  0%]  (Warmup) 
    Chain 3 Iteration:  100 / 4000 [  2%]  (Warmup) 
    Chain 3 Iteration:  200 / 4000 [  5%]  (Warmup) 
    Chain 3 Iteration:  300 / 4000 [  7%]  (Warmup) 
    Chain 3 Iteration:  400 / 4000 [ 10%]  (Warmup) 
    Chain 3 Iteration:  500 / 4000 [ 12%]  (Warmup) 
    Chain 3 Iteration:  600 / 4000 [ 15%]  (Warmup) 
    Chain 3 Iteration:  700 / 4000 [ 17%]  (Warmup) 
    Chain 3 Iteration:  800 / 4000 [ 20%]  (Warmup) 
    Chain 3 Iteration:  900 / 4000 [ 22%]  (Warmup) 
    Chain 3 Iteration: 1000 / 4000 [ 25%]  (Warmup) 
    Chain 3 Iteration: 1100 / 4000 [ 27%]  (Warmup) 
    Chain 3 Iteration: 1200 / 4000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 1300 / 4000 [ 32%]  (Warmup) 
    Chain 3 Iteration: 1400 / 4000 [ 35%]  (Warmup) 
    Chain 3 Iteration: 1500 / 4000 [ 37%]  (Warmup) 
    Chain 3 Iteration: 1600 / 4000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 1700 / 4000 [ 42%]  (Warmup) 
    Chain 3 Iteration: 1800 / 4000 [ 45%]  (Warmup) 
    Chain 3 Iteration: 1900 / 4000 [ 47%]  (Warmup) 
    Chain 3 Iteration: 2000 / 4000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 2001 / 4000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 2100 / 4000 [ 52%]  (Sampling) 
    Chain 3 Iteration: 2200 / 4000 [ 55%]  (Sampling) 
    Chain 3 Iteration: 2300 / 4000 [ 57%]  (Sampling) 
    Chain 3 Iteration: 2400 / 4000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 2500 / 4000 [ 62%]  (Sampling) 
    Chain 3 Iteration: 2600 / 4000 [ 65%]  (Sampling) 
    Chain 3 Iteration: 2700 / 4000 [ 67%]  (Sampling) 
    Chain 3 Iteration: 2800 / 4000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 2900 / 4000 [ 72%]  (Sampling) 
    Chain 3 Iteration: 3000 / 4000 [ 75%]  (Sampling) 
    Chain 3 Iteration: 3100 / 4000 [ 77%]  (Sampling) 
    Chain 3 Iteration: 3200 / 4000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 3300 / 4000 [ 82%]  (Sampling) 
    Chain 3 Iteration: 3400 / 4000 [ 85%]  (Sampling) 
    Chain 3 Iteration: 3500 / 4000 [ 87%]  (Sampling) 
    Chain 3 Iteration: 3600 / 4000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 3700 / 4000 [ 92%]  (Sampling) 
    Chain 3 Iteration: 3800 / 4000 [ 95%]  (Sampling) 
    Chain 3 Iteration: 3900 / 4000 [ 97%]  (Sampling) 
    Chain 3 Iteration: 4000 / 4000 [100%]  (Sampling) 
    Chain 3 finished in 0.2 seconds.
    Chain 4 Iteration:    1 / 4000 [  0%]  (Warmup) 
    Chain 4 Iteration:  100 / 4000 [  2%]  (Warmup) 
    Chain 4 Iteration:  200 / 4000 [  5%]  (Warmup) 
    Chain 4 Iteration:  300 / 4000 [  7%]  (Warmup) 
    Chain 4 Iteration:  400 / 4000 [ 10%]  (Warmup) 
    Chain 4 Iteration:  500 / 4000 [ 12%]  (Warmup) 
    Chain 4 Iteration:  600 / 4000 [ 15%]  (Warmup) 
    Chain 4 Iteration:  700 / 4000 [ 17%]  (Warmup) 
    Chain 4 Iteration:  800 / 4000 [ 20%]  (Warmup) 
    Chain 4 Iteration:  900 / 4000 [ 22%]  (Warmup) 
    Chain 4 Iteration: 1000 / 4000 [ 25%]  (Warmup) 
    Chain 4 Iteration: 1100 / 4000 [ 27%]  (Warmup) 
    Chain 4 Iteration: 1200 / 4000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 1300 / 4000 [ 32%]  (Warmup) 
    Chain 4 Iteration: 1400 / 4000 [ 35%]  (Warmup) 
    Chain 4 Iteration: 1500 / 4000 [ 37%]  (Warmup) 
    Chain 4 Iteration: 1600 / 4000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 1700 / 4000 [ 42%]  (Warmup) 
    Chain 4 Iteration: 1800 / 4000 [ 45%]  (Warmup) 
    Chain 4 Iteration: 1900 / 4000 [ 47%]  (Warmup) 
    Chain 4 Iteration: 2000 / 4000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 2001 / 4000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 2100 / 4000 [ 52%]  (Sampling) 
    Chain 4 Iteration: 2200 / 4000 [ 55%]  (Sampling) 
    Chain 4 Iteration: 2300 / 4000 [ 57%]  (Sampling) 
    Chain 4 Iteration: 2400 / 4000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 2500 / 4000 [ 62%]  (Sampling) 
    Chain 4 Iteration: 2600 / 4000 [ 65%]  (Sampling) 
    Chain 4 Iteration: 2700 / 4000 [ 67%]  (Sampling) 
    Chain 4 Iteration: 2800 / 4000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 2900 / 4000 [ 72%]  (Sampling) 
    Chain 4 Iteration: 3000 / 4000 [ 75%]  (Sampling) 
    Chain 4 Iteration: 3100 / 4000 [ 77%]  (Sampling) 
    Chain 4 Iteration: 3200 / 4000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 3300 / 4000 [ 82%]  (Sampling) 
    Chain 4 Iteration: 3400 / 4000 [ 85%]  (Sampling) 
    Chain 4 Iteration: 3500 / 4000 [ 87%]  (Sampling) 
    Chain 4 Iteration: 3600 / 4000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 3700 / 4000 [ 92%]  (Sampling) 
    Chain 4 Iteration: 3800 / 4000 [ 95%]  (Sampling) 
    Chain 4 Iteration: 3900 / 4000 [ 97%]  (Sampling) 
    Chain 4 Iteration: 4000 / 4000 [100%]  (Sampling) 
    Chain 4 finished in 0.2 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.3 seconds.
    Total execution time: 1.3 seconds.
    



```R
precis(m11.8, depth = 2)
```


<table class="dataframe">
<caption>A precis: 8 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td>-0.4801877</td><td>0.5067051</td><td>-1.2899245</td><td> 0.3076754</td><td>1.004776</td><td>672.8404</td></tr>
	<tr><th scope=row>a[2]</th><td>-0.3831281</td><td>0.5084475</td><td>-1.1874711</td><td> 0.4301130</td><td>1.004671</td><td>675.2622</td></tr>
	<tr><th scope=row>delta[1]</th><td> 1.0608367</td><td>0.5091776</td><td> 0.2648316</td><td> 1.8738063</td><td>1.004677</td><td>679.6456</td></tr>
	<tr><th scope=row>delta[2]</th><td> 1.0173590</td><td>0.5111328</td><td> 0.2165706</td><td> 1.8404276</td><td>1.004617</td><td>689.0876</td></tr>
	<tr><th scope=row>delta[3]</th><td>-0.1991197</td><td>0.5098759</td><td>-1.0096185</td><td> 0.6074035</td><td>1.004674</td><td>676.4325</td></tr>
	<tr><th scope=row>delta[4]</th><td>-0.2309150</td><td>0.5104545</td><td>-1.0402016</td><td> 0.5863615</td><td>1.004494</td><td>682.3512</td></tr>
	<tr><th scope=row>delta[5]</th><td>-0.6735520</td><td>0.5125380</td><td>-1.4800344</td><td> 0.1476060</td><td>1.004700</td><td>683.3706</td></tr>
	<tr><th scope=row>delta[6]</th><td>-2.2300758</td><td>0.5208330</td><td>-3.0608653</td><td>-1.3996685</td><td>1.005397</td><td>697.3947</td></tr>
</tbody>
</table>




```R
par(bg = 'white')
plot(precis(m11.8, depth = 2))
```


    
![png](rethinking_ch11_output_63_0.png)
    


The intercept for male applicants is now a little smaller than for female ones. Let's calculate the contrasts again.


```R
post <- extract.samples(m11.8)
diff_a <- post$a[, 1] - post$a[, 2]
diff_p <- inv_logit(post$a[, 1]) - inv_logit(post$a[, 2])
precis(list(diff_a=diff_a, diff_p=diff_p))
```


<table class="dataframe">
<caption>A precis: 2 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>diff_a</th><td>-0.09705960</td><td>0.0813917</td><td>-0.22777864</td><td>0.03307494</td><td>▁▁▁▂▅▇▇▅▂▁▁▁▁</td></tr>
	<tr><th scope=row>diff_p</th><td>-0.02201809</td><td>0.0187732</td><td>-0.05237983</td><td>0.00762094</td><td>▁▁▂▇▇▂▁▁     </td></tr>
</tbody>
</table>



Why did adding the department in make such a change, where now the male admission rate is about 2% lower than the female one? Again, the departments had wildly varying admission rates. Let's take a look.


```R
pg <- with(dat_list,
    sapply(1:6, function(k) {
        applications[dept_id == k]/sum(applications[dept_id == k])
    })
)
rownames(pg) <- c("Male", "Female")
colnames(pg) <- unique(d$dept)
round(pg, 2)
```


<table class="dataframe">
<caption>A matrix: 2 × 6 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>A</th><th scope=col>B</th><th scope=col>C</th><th scope=col>D</th><th scope=col>E</th><th scope=col>F</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Male</th><td>0.88</td><td>0.96</td><td>0.35</td><td>0.53</td><td>0.33</td><td>0.52</td></tr>
	<tr><th scope=row>Female</th><td>0.12</td><td>0.04</td><td>0.65</td><td>0.47</td><td>0.67</td><td>0.48</td></tr>
</tbody>
</table>



These are the proportions of applications from men / women across the departments. As you can see, there is quite the wide range!

The department is probably a confound, in that it misleads us about the direct causal effect. However, it *isn't* a confound in the sense that it is probably a genuine causal path. In DAG form:


```R
library(dagitty)

par(bg = 'white')
dag <- dagitty("dag{G -> D -> A; G -> A}")
plot(dag)
```

    Plot coordinates for graph not supplied! Generating coordinates, see ?coordinates for how to set your own.
    



    
![png](rethinking_ch11_output_69_1.png)
    


where here $G$ is the gender, $D$ is the department, and $A$ is the acceptance rate.

In addition to the direct effect of $G$ on $A$, there is an indirect causal path through the department $D$. In order to find $G \to A$, we need to close that path by conditioning on $D$. Model `m11.8` does this.

This is an example of **Mediation Analysis**.

However, we shouldn't stop there. Our DAG might be wrong. What if there is another unobserved confound that is affecting both the department and the admissions rate?


```R
library(dagitty)

dag <- dagitty('dag{
    G[pos="0, 0"]
    A[pos="1, 0"]
    D[pos="0.5, -0.5"]
    U[latent,pos="1,-0.5"]
    G -> D -> A; G -> A; D <- U -> A}')
par(bg = 'white')
drawdag(dag)
```


    
![png](rethinking_ch11_output_71_0.png)
    


That unobserved $U$ could be a host of things - maybe academic ability? Note this if this DAG is correct, then by conditioning on $D$ we are really conditioning on a collider, opening up a non-causal path between $G$ and $A$: $G \to D \leftarrow U \to A$.

## 2 Poisson Regression

Binomial GLMs should be used when the count is from 0 to some upper bound that we know about. However, sometimes we don't know the upper bound; in these cases, we should use a Poisson regression. For instance, what if we go fishing and return with 17 fish? It is unclear in this case what the theoretical upper bound would be.

Recall that the Poisson is a special case of the binomial where $n$ is very large and $p$ is very small. Then the mean and variance become basically the same, and we get the **Poisson Distribution**.

The model with a Poisson distribution is even simpler, since there is just one parameter: $\lambda$, the expected average rate of events.

$$
y_i \sim \text{Poisson}(\lambda)
$$

Since $\lambda$ needs to be positive, we conventionally use a log link function here

$$
\begin{align*}
y_i &\sim \text{Poisson}(\lambda_i) \\
\log(\lambda_i) &= \alpha_i + \beta (x_i - \bar{x})
\end{align*}
$$

One thing to keep in mind is that since we are using the log link, it implies that the rate scales exponentially with the predictor variable, which may be unrealistic.

### 2.1 Example: Oceanic tool complexity

Different island societies in Oceania provide natural experiments in technological evolution. Different islands had many different types of tools. One theory is that larger populations will produce more complex tool kits.


```R
data(Kline)
d <- Kline
d
```


<table class="dataframe">
<caption>A data.frame: 10 × 5</caption>
<thead>
	<tr><th scope=col>culture</th><th scope=col>population</th><th scope=col>contact</th><th scope=col>total_tools</th><th scope=col>mean_TU</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Malekula  </td><td>  1100</td><td>low </td><td>13</td><td>3.2</td></tr>
	<tr><td>Tikopia   </td><td>  1500</td><td>low </td><td>22</td><td>4.7</td></tr>
	<tr><td>Santa Cruz</td><td>  3600</td><td>low </td><td>24</td><td>4.0</td></tr>
	<tr><td>Yap       </td><td>  4791</td><td>high</td><td>43</td><td>5.0</td></tr>
	<tr><td>Lau Fiji  </td><td>  7400</td><td>high</td><td>33</td><td>5.0</td></tr>
	<tr><td>Trobriand </td><td>  8000</td><td>high</td><td>19</td><td>4.0</td></tr>
	<tr><td>Chuuk     </td><td>  9200</td><td>high</td><td>40</td><td>3.8</td></tr>
	<tr><td>Manus     </td><td> 13000</td><td>low </td><td>28</td><td>6.6</td></tr>
	<tr><td>Tonga     </td><td> 17500</td><td>high</td><td>55</td><td>5.4</td></tr>
	<tr><td>Hawaii    </td><td>275000</td><td>low </td><td>71</td><td>6.6</td></tr>
</tbody>
</table>



This data show counts and amount of inter-island conflict for 10 different islands in Oceania (historical data). We'll use `total_tools` as the outcome and model
1. Number of tools increases with the log of population size (mostly because we care about the *order of magnitude of the population)
1. Number of tools increases with `contact`
1. Impact of population on tool counts is moderated by high `contact`. That is, the association between the `total_tools` and hte log `poulation` depends on `contact`; we'll look for a positive interaction between log `population` and `contact` rate.


```R
d$P <- scale(log(d$population))
d$contact_id <- ifelse(d$contact == "high", 2, 1)
```

$$
\begin{align*}
T_i &\sim \text{Poisson}(\lambda_i) \\
\log (\lambda_i) &= \alpha_{\text{CID}[i]} + \beta_{\text{CID}[i]} \log P_i \\
\alpha_j &\sim \text{TBD} \\
\beta_j &\sim \text{TBD} \\
\end{align*}
$$

As with before, we need to do some experimentation to find sensible priors. To get an idea, let's start with a simpler model.

$$
\begin{align*}
T_i &\sim \text{Poisson}(\lambda_i) \\
\log \lambda_i &= \alpha \\
\alpha &\sim \text{Normal}(0, 10)
\end{align*}
$$

If $\alpha$ has a normal distribution, then $\lambda$ has a log-normal distribution. So, let's plot that!


```R
x <- seq(0, 100, length.out = 200)
y <- dlnorm(x, 0, 10)
ggplot(data.frame(x=x, y=y), aes(x, y)) +
    geom_line()
```


    
![png](rethinking_ch11_output_78_0.png)
    


This doesn't work - it has a huge spike at 0 and and average of about 1e12 unique tool types per island, which is clearly absurd. One way to fix this is to shift the mean above zero, since the log scale will transform anything below 0 to be between 0 and 1 on the new scale.


```R
x <- seq(0, 100, length.out = 2000)
y1 <- dlnorm(x, 0, 10)
y2 <- dlnorm(x, 3, 0.5)
ggplot(rbind(data.frame(x=x, y=y1, label="a ~ dnorm(0, 10)"), data.frame(x=x, y=y2, label="a ~ dnorm(3, 0.5)")), aes(x, y, group = label, colour = label)) +
    geom_line()

```


    
![png](rethinking_ch11_output_80_0.png)
    


The mean is now about 20, which is definitely more plausible.

What about the prior for $\beta$? Let's use our existing prior for $\alpha$ and simulate a bunch of different population effects.


```R
N <- 100
a <- rnorm(N, 3, 0.5)
b <- rnorm(N, 0, 10)

x <- seq(-2, 2, length.out = 201)
plot_df <- data.frame(x = numeric(), y = numeric(), n = integer())

for (i in 1:N) {
    y <- exp(a[i] + b[i] * x)
    interim_df <- data.frame(
        x = x,
        y = y,
        n = i
    )
    plot_df <- rbind(plot_df, interim_df)
}

ggplot(
    plot_df, aes(x, y, group = n)
) +
    geom_line() +
    labs(
        x = 'Log Population',
        y = "Total Tools",
        title = "b ~ dnorm(0, 10)"
    ) +
    ylim(c(0, 100))
```

    Warning message:
    “[1m[22mRemoved 7985 rows containing missing values (`geom_line()`).”



    
![png](rethinking_ch11_output_82_1.png)
    


Again, these priors are not great. They all tend to think that either the growth will be explosive around the population mean or ecline precipitously around the same. Neither of these are all that plausible.

After some experimentation, let's try $\beta \sim \text{Normal}(0, 0.2)$:


```R
N <- 100
a <- rnorm(N, 3, 0.5)
b <- rnorm(N, 0, 0.2)

x <- seq(-2, 2, length.out = 201)
plot_df <- data.frame(x = numeric(), y = numeric(), n = integer())

for (i in 1:N) {
    y <- exp(a[i] + b[i] * x)
    interim_df <- data.frame(
        x = x,
        y = y,
        n = i
    )
    plot_df <- rbind(plot_df, interim_df)
}

ggplot(
    plot_df, aes(x, y, group = n)
) +
    geom_line() +
    labs(
        x = 'Log Population',
        y = "Total Tools",
        title = "b ~ dnorm(0, 0.2)"
    ) +
    ylim(c(0, 100))
```

    Warning message:
    “[1m[22mRemoved 86 rows containing missing values (`geom_line()`).”



    
![png](rethinking_ch11_output_84_1.png)
    


These priors seem much more reasonable. There is still the poosibility of strong relationships, but most of the results are pretty flat, which is what we are looking for.

These priors can also be a bit hard to interpret because we are using a normalized scale, not the actual population or log population. Let's see what we get if we use the more natural scale:


```R
x_seq <- seq(from = log(100), to = log(200000), length.out = 100)
# this is a matrix where each row corresponds to the y values for that x
lambda <- sapply(x_seq, function(x) exp(a + b * x))

plot_df <- data.frame(x = numeric(), y = numeric(), a = numeric(), b = numeric())

for (i in 1:N) {
    interim_df <- data.frame(
        x = x_seq,
        y = exp(a[i] + b[i] * x_seq),
        a = rep(a[i], length(x_seq)),
        b = rep(b[i], length(x_seq))
    )
    plot_df <- rbind(plot_df, interim_df)
}

ggplot(plot_df, aes(x, y, group = interaction(a, b))) +
    geom_line() +
    ylim(c(0, 500)) +
    labs(x = 'log population', y = 'Total tools', title = "a ~ dnorm(3, 0.5), b ~ dnorm(0, 0.2)")
```

    Warning message:
    “[1m[22mRemoved 583 rows containing missing values (`geom_line()`).”



    
![png](rethinking_ch11_output_86_1.png)
    


This looks much better! We can also plot against the raw population:


```R
raw_population_plot_df <- plot_df
raw_population_plot_df$x <- exp(raw_population_plot_df$x)

ggplot(raw_population_plot_df, aes(x, y, group = interaction(a, b))) +
    geom_line() +
    ylim(c(0, 500)) +
    labs(x = 'log population', y = 'Total tools', title = "a ~ dnorm(3, 0.5), b ~ dnorm(0, 0.2)")
```

    Warning message:
    “[1m[22mRemoved 583 rows containing missing values (`geom_line()`).”



    
![png](rethinking_ch11_output_88_1.png)
    


Note that now the curves bend in the opposite direction! **Poisson models** with a log link create log-linear relationships with the predictor variables. If the predictor variable is itself logged, then that means that we are assuming a *diminishing rate of return* for that variable. In our case, each additional person contributes a smaller increase in the expected number of tools.

This pattern is true of lots of things, which is one of the advantages of using logarithms.

Now let's code two models: one simple one (intercept-only), and the other with both parameters.


```R
dat <- list(
    T = d$total_tools,
    P = d$P,
    cid = d$contact_id
)

# intercept only
m11.9 <- ulam(
    alist(
        T ~ dpois(lambda),
        log(lambda) <- a,
        a ~ dnorm(3, 0.5)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)

# interaction model
m11.10 <- ulam(
    alist(
        T ~ dpois(lambda),
        log(lambda) <- a[cid] + b[cid] * P,
        a[cid] ~ dnorm(3, 0.5),
        b[cid] ~ dnorm(0, 0.2)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)

```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.
    
    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.
    



```R
compare(m11.9, m11.10, func = PSIS)
```

    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    
    Some Pareto k values are very high (>1). Set pointwise=TRUE to inspect individual points.
    



<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>PSIS</th><th scope=col>SE</th><th scope=col>dPSIS</th><th scope=col>dSE</th><th scope=col>pPSIS</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m11.10</th><td> 85.44932</td><td>13.26048</td><td> 0.00000</td><td>      NA</td><td>7.031250</td><td>1.00000e+00</td></tr>
	<tr><th scope=row>m11.9</th><td>139.54714</td><td>32.87346</td><td>54.09782</td><td>31.93818</td><td>6.913467</td><td>1.78981e-12</td></tr>
</tbody>
</table>



Unsurprisingly, m11.10 does a better job than the intercept-only model m11.9. What is surprising is that the effective number of parameters pPSIS is *larger* for the model with only a single parameter. This is due to the link function. Once we go non-linear, then we stop being able to easily interpret pPSIS. Basically, parameters near the cut-off point contribute less overfitting than those far from the boundary, which has some surprising effects on pPSIS.

Bottom line: overfitting risk depends on *both* the structural complexity of the model and the composition of the sample.

In this model, a major source of overfitting risk is in the points that were flagged by PSIS. Let's examine them!


```R
k <- PSIS(m11.10, pointwise = TRUE)$k

# setting up the predictions
ns <- 100
P_seq <- seq(from = -1.4, to = 3, length.out = ns)

#low contact (cid = 1) predictions
lambda <- link(m11.10, data = data.frame(P = P_seq, cid = 1))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
low_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 1
)

# high contact (cid = 2) predictions
lambda <- link(m11.10, data = data.frame(P = P_seq, cid = 2))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
high_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 2
)

predictions <- rbind(low_contact_df, high_contact_df)
predictions$cid <- factor(predictions$cid)

plot_df <- data.frame(P = dat$P, T = dat$T, cid = factor(dat$cid), k = k)
ribbon_df <- data.frame(P = predictions$P, lower = predictions$lower, upper = predictions$upper, cid = factor(predictions$cid))
ggplot() +
    geom_point(data = plot_df, mapping = aes(x = P, y = T, shape = cid, size = 1 + normalize(k))) +
    geom_line(data = predictions, mapping = aes(P, mu, linetype = cid)) +
    geom_ribbon(data = ribbon_df, mapping = aes(x = P, ymin = lower, ymax = upper, group = cid), alpha = 0.2)  +
    coord_cartesian(ylim = c(0, 75))
```

    Some Pareto k values are very high (>1). Set pointwise=TRUE to inspect individual points.
    



    
![png](rethinking_ch11_output_93_1.png)
    


We can also do the same, but on the regular population scale, just by 'un-converting' `P`


```R
# setting up the predictions
ns <- 100
P_seq <- seq(from = -5, to = 3, length.out = ns)
pop_seq <- exp(P_seq * 1.53 + 9) # 1.53 is the sd of log (population); 9 is the mean

#low contact (cid = 1) predictions
lambda <- link(m11.10, data = data.frame(P = P_seq, cid = 1))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
low_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 1
)

# high contact (cid = 2) predictions
lambda <- link(m11.10, data = data.frame(P = P_seq, cid = 2))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
high_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 2
)

predictions <- rbind(low_contact_df, high_contact_df)
predictions$cid <- factor(predictions$cid)
predictions$population <- pop_seq

plot_df <- data.frame(population = d$population, total_tools = d$total_tools, cid = factor(dat$cid), k = k, culture = d$culture)
ribbon_df <- data.frame(population = pop_seq, lower = predictions$lower, upper = predictions$upper, cid = factor(predictions$cid))
ggplot() +
    geom_point(data = plot_df, mapping = aes(x = population, y = total_tools, shape = cid, size = 1 + normalize(k))) +
    geom_text(data = plot_df, mapping = aes(x = population, y = total_tools, label = culture), hjust = 0, vjust = 2) +
    geom_line(data = predictions, mapping = aes(population, mu, linetype = cid)) +
    geom_ribbon(data = ribbon_df, mapping = aes(x = population, ymin = lower, ymax = upper, group = cid), alpha = 0.2) + 
    coord_cartesian(xlim = c(0, 300000), ylim = c(0, 75))
```


    
![png](rethinking_ch11_output_95_0.png)
    


There are a few points which are influential; Hawaii, Tonga, and Tap. We will deal with these later.

Also note that the model is quite silly; the high-contact line actually passes *under* the low-contact one. This is largely due to lack of data; there aren't any high-contact, large population societies (Hawaii is low-contact). But this still shouldn't happen; if we imagine a counterfactual where Hawaii was high-contact, we wouldn't expect the number of tools to go *down*.

Largely this is because we let the intercept be a free parameter. But that is incorrect! If we have 0 population, we know that we have 0 tools. We get this for free, and should probably integrate it into the model.

Instead of the GLM we created above, we could create a model which is constructured from our scientific understanding of *how* population and contact influence tool use. Maybe something that takes into account the fact that new tools are gradually developed, &c. One way of accounting for this might be to model some rate of innovation and take into account how each person has the potential to innovate, but with diminishing returns.

When we use this model we get better performance than the 'geocentric' GLM above, according to WAIC and PSIS. More importantly, the behaviour better reflects what we know to be true.

### 2.2 Negative binomial (gamma-Poisson) models

Often a lot of unexplained variance in a Poisson model. Presumably this is because the rate varies from case to case. A very common extension of the Poisson GLM is to change the Poisson distribution to a **Negative Binomial** distribution. This is a special case of the Poisson (and is sometimes called the **Gamma-Poisson** distribution). It is a mixture of different Poisson distributions, in the same way that the **Student t** distribution is a mixture of **Normal distributions**.

### 2.3 Example: Exposure and the offset

For a **Poisson distribution**, $\lambda$ can be thought of as the average number of events or as the rate; the two are equivalent. This lets us make models for which the exposure varies. For instance, maybe you have two different monastaries. From one you have the daily total of manuscripts created, and from the other you have the daily total. How can you reconcile these?

Implicitly, we can think of $\lambda$ being equal to the average number of events, $\mu$, per time unit, $\tau$. Thus, $\lambda = \mu / \tau$, so we can redo the link function as

$$
\begin{align*}
y_i &\sim \text{Poisson}(\lambda_i) \\
\log \lambda_i = \log \frac{\mu_i}{\tau_i} = \alpha + \beta x_i
\end{align*}
$$

We can also rewrite this as

$$
\log \lambda_i = \log \mu_i - \log \tau_i = \alpha + \beta x_i
$$

The $\tau$ values are the different exposures. If different observations have different exposures, then the expected value on row $i$ is

$$
\log \mu_i = \log\tau_i + \alpha + \beta x_i
$$

When $\tau_i = 1$, then $\log \tau_i = 0$ and we just have the regular Poisson GLM. But if the exposure varies from case to case, then $\tau_i$ scales how many events we expect to see.

$$
\begin{align*}
y_i &\sim \text{Poisson}(\mu_i) \\
\log \mu_i &= \log \tau_i + \alpha + \beta x_i
\end{align*}
$$

The $\log \tau$ term that we added is typically called the *offset*.

Let's look at an example using simulated data to ensure that we are getting the correct values back.

We own a monastery! The true rate of producting manuscripts is $\lambda = 1.5$ per day.


```R
num_days <- 30
true_lambda <- 1.5
y <- rpois(num_days, true_lambda)
```

Now you're considering buying a new monastery, but this one keeps weekly totals. It turns out (although we don't know it) that the true rate at the new monastery is 0.5 per day.


```R
num_weeks <- 4
new_true_lambda <- 0.5
y_new <- rpois(num_weeks, new_true_lambda * 7)
```


```R
y_all <- c(y, y_new)
exposure <- c(rep(1, num_days), rep(7, num_weeks))
monastery <- c(rep(0, num_days), c(rep(1, num_weeks)))
d <- data.frame(y = y_all, days = exposure, monastery = monastery)

d
```


<table class="dataframe">
<caption>A data.frame: 34 × 3</caption>
<thead>
	<tr><th scope=col>y</th><th scope=col>days</th><th scope=col>monastery</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>2</td><td>1</td><td>0</td></tr>
	<tr><td>3</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>2</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>3</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>2</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>2</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>3</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>3</td><td>1</td><td>0</td></tr>
	<tr><td>2</td><td>1</td><td>0</td></tr>
	<tr><td>0</td><td>1</td><td>0</td></tr>
	<tr><td>3</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>2</td><td>7</td><td>1</td></tr>
	<tr><td>4</td><td>7</td><td>1</td></tr>
	<tr><td>4</td><td>7</td><td>1</td></tr>
	<tr><td>8</td><td>7</td><td>1</td></tr>
</tbody>
</table>




```R
# create the offset (log of the exposure)
d$log_days <- log(d$days)

m11.12 <- quap(
    alist(
        y ~ dpois(lambda),
        log(lambda) <- log_days + a + b * monastery,
        a ~ dnorm(0, 1),
        b ~ dnorm(0, 1)
    ),
    data = d
)
```

To compute the posterior distributions of $\lambda$ in each monastery, we sample form the posterior and then use the linear model, but without the offset. The offset is not used during predictions because the parameters are already on the daily scale.


```R
post <- extract.samples(m11.12)
lambda_old <- exp(post$a)
lambda_new <- exp(post$a + post$b)
precis(data.frame(lambda_old, lambda_new))
```


<table class="dataframe">
<caption>A precis: 2 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>lambda_old</th><td>1.2223943</td><td>0.1981874</td><td>0.9329753</td><td>1.5619908</td><td>▁▂▇▇▂▁▁▁      </td></tr>
	<tr><th scope=row>lambda_new</th><td>0.6794961</td><td>0.1553942</td><td>0.4628275</td><td>0.9504016</td><td>▁▁▂▇▇▅▃▁▁▁▁▁▁▁</td></tr>
</tbody>
</table>




```R
par(bg = 'white')
plot(
    precis(data.frame(lambda_old, lambda_new))
)
```


    
![png](rethinking_ch11_output_105_0.png)
    


So it looks like this method works just like we expect!

## 3 Multinomial and categorical variables

The binomial works when there are only two outcomes, but in general, more than two things can happen! For instance, recally that in **Chapter 2** we were drawing blue and white marbles from the bag. Since there were only two types, we used the binomial distribution. Now suppose that we add red marbles to the bag. Now we can't! We have to use the **Multinomial Distribution** instead.

The probability distribution for the **multinomial** looks a lot like for the binomial. If we $K$ types of events and each has probability $p_1, p_2, \ldots, p_K$, then the probability of seeing $y_1$ of type 1, $y_2$ of type 2, $\ldots$ $y_K$ of type $K$ is 

$$
\text{P}(y_1, y_2, \dots, y_K | n, p_1, \dots, p_K) = \frac{n!}{\Pi_i y_i!}\Pi_{i=1}^K p_i^{y_i}
$$

A model built on the **multinomial** is called a **Categorial regression**, usually when each event is expressed on one row (like in logistic regression). In machine learning, this model is sometimes known as the **maximum entropy classifier**. It's complicated to build a model, since as the number of events increases, so do the number of choices that we have to make about modelling those events.

There are two main approaches. One uses a generalization of the **logit link** (we'll call this the *explicit* approach). The second transforms the **multinomial** likelihoods into a series of **Poisson** likelihoods.

The natural (and conventional) link is the **Multinomial logit**, also known as the **Softmax function**. This link takes a vector of scores, one for each of the $K$ event types, and computes the probability of a particular type of event $k$ as

$$
\text{Pr}(k | s_1, s_2, \dots, s_k) = \frac{e^{s_k}}{\sum_{i=1}^{K}e^{s_i}}
$$

This type of model (with the `softmax` link) is called the **Multinomial logistic regression**.

The biggest issue is what to do with the multiple linear models. In the **binomial** GLM, we just had to pick one of the two outcomes and build a linear model for its log-odds; then the other one was handled automatically. In a **multinomial** GLM, we need $K-1$ models for the $K$ events. One of the outcomes is chosen as the *pivot* (that's the one that we won't model explicitly). In the other $K-1$ linear models, we can pick whatever predictor variables we want. In general, they won't be the same for the different event types.

There are two basic cases:
1. Predictors have different values for different values of the outcome, and
1. Parameters are distinct for each value of the outcome.

The first case is useful when each type of event has its own quantitative traits and you want to estimate the association between those trains and the proability each type of event appears in the data.

The second case is useful when you are interested instead in features of some entity that produces each event, whatever type it turns out to be.

We'll consider each case separately and go through an example of each. You can mix both types in the same model, but it'll be clearer if we keep them distinct for now.

### 3.1 Predictor matched to outcomes

We're modelling choice of careers for young adults. One of the predictors will be expected income. In that case, $\beta_{\text{income}}$ appears in each linear model, but a *different* income value multiplies the parameter in each linear model.

Here's a simulation in R. We are simulating career choice between three different careers, each of which has its own income trait. This is then used to assign a score to each type of event. When the model is fit to the data, one of the scores is held constant and the other two are estimated.


```R
N <- 500 # number of individuals
income <- c(1, 2, 5)
score <- 0.5 * income

# convert scores to probabilities
p <- softmax(score[1], score[2], score[3])

# now simulate the choices
# outcome career holds event type values, not counts
career <- rep(NA, N)

# randomly choose the career for each individual
set.seed(34302)
for (i in 1:N) {
    career[i] <- sample(1:3, size = 1, prob = p)
}
```

To fit the model, we use the `dcategorical` likelihood, which is the multinomial logistic regression distribution. It works when each value in the outcome variable (`career`) contains the individual event type on each row.

To convert the scores to probabilities, we'll use the multinomial logit link (`softmax`). Then each career gets its own linear model. There are no intercepts in the simulation above, but if income doesn't predict career choice, you still want an intercept to account for differences in frequency.


```R
# Doing it in Stan!

code_m11.13 <- "
data {
    int N; // number of individuals
    int K; // number of careers
    array[N] int career; // outcome
    vector[K] career_income;
}
parameters {
    vector[K-1] a; // intercepts
    real<lower=0> b; // association of income with choice
}
model {
    vector[K] p;
    vector[K] s;
    a ~ normal(0, 1);
    b ~ normal(0, 0.5);
    s[1] = a[1] + b * career_income[1];
    s[2] = a[2] + b * career_income[2];
    s[3] = 0; // pivot
    p = softmax(s);
    career ~ categorical(p);
}
"

dat_list <- list(
    N = N,
    K = 3,
    career = career,
    career_income = income
)
m11.13 <- stan(model_code = code_m11.13, data = dat_list, chains = 4)
```

    Running MCMC with 4 sequential chains...
    
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
    Chain 1 finished in 0.1 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.1 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.1 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.1 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.5 seconds.
    


    Warning: 1 of 2000 (0.0%) transitions ended with a divergence.
    See https://mc-stan.org/misc/warnings for details.
    
    



```R
precis(m11.13, 2)
```


<table class="dataframe">
<caption>A precis: 3 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>n_eff</th><th scope=col>Rhat4</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td>-2.1308811</td><td>0.1885815</td><td>-2.438134550</td><td>-1.8439118</td><td>503.6699</td><td>1.011036</td></tr>
	<tr><th scope=row>a[2]</th><td>-1.7869542</td><td>0.2545804</td><td>-2.252708600</td><td>-1.4450158</td><td>357.8658</td><td>1.015904</td></tr>
	<tr><th scope=row>b</th><td> 0.1317465</td><td>0.1149318</td><td> 0.006917019</td><td> 0.3559642</td><td>316.8830</td><td>1.020361</td></tr>
</tbody>
</table>




```R
par(bg = 'white')
plot(precis(m11.13, 2))
```


    
![png](rethinking_ch11_output_113_0.png)
    


It's very difficult to figure out what these actually mean. We arbitrarily chose the third career to be the pivot, but we could have chosen any of them. In that case the actual numbers we got out would be different but the proedictions would have been the same.

To conduct a counterfactual experiment, we can extract the samples and make our own. We want to compare the counterfactual where the income is changed. Let's imagine doubling the income of career 2.


```R
post <- extract.samples(m11.13)

# set up the logit scores
s1 <- with(post, a[, 1] + b * income[1])
s2_orig <- with(post, a[, 2] + b * income[2])
s2_new <- with(post, a[, 2] + b * income[2] * 2) # doubling the income

# compute probabilities for original and counterfactual
p_orig <- sapply(1:length(post$b), function(i) {
    softmax(c(s1[i], s2_orig[i], 0))
})
p_new <- sapply(1:length(post$b), function(i) {
    softmax(c(s1[i], s2_new[i], 0))
})

p_diff <- p_new[2, ] - p_orig[2, ]
precis(p_diff)
```


<table class="dataframe">
<caption>A precis: 1 × 5</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>histogram</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>p_diff</th><td>0.04145759</td><td>0.04061757</td><td>0.001883568</td><td>0.1216541</td><td>▇▂▁▁▁▁</td></tr>
</tbody>
</table>



So on average, a 4% increase in career choice when the income is doubled. (NB this number is different in the textbook - I'm not sure where I went wrong as the code is the same...). Note that this is dependent on the other careers; if they were different then this number would be as well. That's just how choices work - if you change the alternatives then you change the choice.

### 3.2 Predictors matched to observations

Now consider a model where each observed outcome has unique predictor values. We'll still model career choice, but now we want to estimate the association between each person's family income and the career they choose. The predictor variable must have the same value in linear linear model, for each row of data. But now there's is a unique parameter multiplying it in each linear model. This provides an estimate of the impact of family income on choice, for each type of career.


```R
N <- 500

# simulate family income for each student
family_income <- runif(N)

# assign a unique coefficient to each event
b <- c(-2, 0, 2)
career <- rep(NA, N)
for (i in 1:N) {
    score <- 0.5 * (1:3) + b * family_income[i] # now we're accounting for family income
    p <- softmax(score[1], score[2], score[3])
    career[i] <- sample(1:3, size = 1, prob = p)
}

code_m11.14 <- "
    data {
        int N; // observations
        int K; // careers
        array[N] int career; // the actual careers of the individuals
        array[N] real family_income; // the family income for each student
    }
    parameters {
        vector[K-1] a; // intercepts
        vector[K-1] b; // coefficients on family income
    }
    model {
        vector[K] p;
        vector[K] s;
        a ~ normal(0, 1.5);
        b ~ normal(0, 1);
        for ( i in 1:N ) {
            for (j in 1:(K-1)) {
                s[j] = a[j] + b[j] * family_income[i];
            }
            s[K] = 0; // pivot
            p = softmax(s);
            career[i] ~ categorical(p);
        }
    }
"

dat_list <- list(N = N, K = 3, career = career, family_income = family_income)
m11.14 <- stan(model_code = code_m11.14, data = dat_list, chains = 4)
```

    Running MCMC with 4 sequential chains...
    
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
    Chain 1 finished in 2.3 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 2.3 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 2.5 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 2.4 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 2.4 seconds.
    Total execution time: 9.7 seconds.
    



```R
precis(m11.14, 2)
```


<table class="dataframe">
<caption>A precis: 4 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>n_eff</th><th scope=col>Rhat4</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a[1]</th><td>-0.7744673</td><td>0.2509515</td><td>-1.194781</td><td>-0.3826142</td><td>898.6792</td><td>1.0026843</td></tr>
	<tr><th scope=row>a[2]</th><td>-0.7185261</td><td>0.2137498</td><td>-1.064691</td><td>-0.3804557</td><td>957.8916</td><td>1.0005249</td></tr>
	<tr><th scope=row>b[1]</th><td>-3.4570791</td><td>0.5535833</td><td>-4.330678</td><td>-2.5580449</td><td>914.3921</td><td>1.0024679</td></tr>
	<tr><th scope=row>b[2]</th><td>-1.2170905</td><td>0.3726337</td><td>-1.817545</td><td>-0.6312219</td><td>983.5261</td><td>0.9998387</td></tr>
</tbody>
</table>



Again, these are difficult to interpret. Probably the best way is to compute the implied predictions when you change the family income.

### 3.3 Multinomial in disguise as Poisson

Another way to fit a multinomial / categorical model is to refactor it into a series of **Poisson** likelihoods. It sounds crazy, but is actually both common and pricipled to do it this way (it's justified by the mathematics and usually faster / easier to compute). Here's an example!

We'll use the binomial (special case of multinomial) and re do it as a **Poisson**. In this case, let's re-use the UC Berkeley admissions data.


```R
data(UCBadmit)
d <- UCBadmit

# binomial model
m_binom <- quap(
    alist(
        admit ~ dbinom(applications, p),
        logit(p) <- a,
        a ~ dnorm(0, 1.5)
    ),
    data=d
)

dat <- list(admit = d$admit, rej = d$reject)
m_poisson <- ulam(
    alist(
        admit ~ dpois(lambda1),
        rej ~ dpois(lambda2),
        log(lambda1) <- a1,
        log(lambda2) <- a2,
        c(a1, a2) ~ dnorm(0, 1.5)
    ),
    data = dat,
    chains = 3,
    cores = 3
)
```

    Running MCMC with 3 parallel chains, with 1 thread(s) per chain...
    
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
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 1 finished in 0.0 seconds.
    Chain 2 finished in 0.0 seconds.
    Chain 3 finished in 0.0 seconds.
    
    All 3 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.2 seconds.
    



```R
# look at the posterior means
inv_logit(coef(m_binom))
```


<strong>a:</strong> 0.387790714248965


In the Poisson model, the implied probability of admission is

$$
P_{\text{admit}} = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{e^{a_1}}{e^{a_1} + e^{a_2}}
$$


```R
k <- coef(m_poisson)
a1 <- k['a1']
a2 <- k['a2']
exp(a1) / (exp(a1) + exp(a2))
```


<strong>a1:</strong> 0.38763726805142


So we got the same answer both times! Normally we use the categorical distribution, but sometimes it's easier to do the Poisson (or you may encounter it in the wild), so it's good to know.

## 4 Practice

### 4.1 Easy

### 4.2 11E1 If an event has probability 0.35, what are the log-odds of the event?


```R
p <- 0.35
log(p / (1 - p))
```


-0.619039208406224


### 4.3 11E2 If an event has log-odds 3.2, what is the probability of the event?

$$
\begin{align*}
3.2 &= \log \frac{p}{1 - p} \\
e^{3.2} &= \frac{p}{1 - p} \\
e^{3.2} - e^{3.2}p &= p \\
\frac{e^{3.2}}{1 + e^{3.2}} &= p
\end{align*}
$$


```R
exp(3.2) / (1 + exp(3.2))
```


0.960834277203236


### 4.4 11E3 Suppose that a coefficient in a logistic regression has value 1.7. What does this imply about the proportional change in odds of the outcome?

In a logistic regression, exponentiating the coefficient tells us the proportional change in odds.


```R
exp(1.7)
```


5.4739473917272


This implies that the proportional change in odds will be a five-fold increase for every increase in the underlying predictor. (Think analogously to the slope of a regression line: a one-unit change in x is associated with a ???? difference in y).

### 4.5 11E4 Why do Poisson regressions sometimes require the use of an *offset*? Provide an example.

They sometimes require an offset when there is a differing amount of exposure to the underlying causal phenomenon. For instance, we might be measuring the number of butterflies we see along a trail, but one person is measuring every 100m and the other every km. In a Poisson, $\lambda$ is the expected number of observations per unit of whatever (maybe in this case, $\lambda$ is 'butterflies per metre'). If the units are different, we can think of

$$
\lambda = \frac{\mu}{\tau}
$$

Where $\mu$ is the expected number of observations (recorded number per 100m or 1km, depending) and $\tau$ is the actual time / distance we are measuring for that particular observation (100m or 1km).

In a Poisson we generally use the log link, so then we have

$$
\log \lambda = \log \frac{\mu_i}{\tau_i} = \alpha + \beta x
$$

We really care about $\mu_i$, the normalized rate, so then we have

$$
\log \mu_i = \alpha + \beta x + \underbrace{\log \tau_i}_{\text{offset}}
$$

Basically, the offset accounts for the differing exposures across the observations.

### 4.6 11M1 As explained in the chapter, binomial data can be organized in aggregated and disaggregated forms without any influence on inference. However, the likellihood of the data des change when the data are converted between the two formats. Can you explain why?

Essentially, when you aggregate data you lose information --- specifically, you lose information about the order in which they occurred. This means that the likelihood of the aggregated data is higher than for the disaggregated one. For instance, rolling dice and getting a 1, then a 2 has probability 1/6 * 1/6 = 1 / 36. However, if we aggregate the data and just say that we rolled the dice and got a 1 and a 2 in some order, then we have a likelihood of 2 * 1/ 6 * 1 / 6 (since there are two orders that we could have rolled them in).

### 4.7 11M2 If a coefficient in a Poisson regression has a value of 1.7, what does this imply about the change in the outcome?

$$
\begin{align*}
y_i &\sim \text{Poisson}(\lambda_i) \\
\log \lambda_i &= \alpha + \beta x_i \\
\end{align*}
$$

if $\beta = 1.7$, then let's look at $x_i$ against $x_i + 1$.

$$
\begin{align*}
\log \lambda_{i, +1} - \log \lambda_i &= \alpha + \beta( x_i + 1) - (\alpha + \beta x_i) \\
    &= \beta \\
\log \frac{\lambda_{i, +1}}{\lambda_i} &= \beta \\
\frac{\lambda_{i, +1}}{\lambda_i} &= e^{\beta} \\
\lambda_{i, +1} &= e^\beta \lambda_i \\
    &= e^{1.7} \lambda_i \\
    &\approx 5.5 \lambda_i
\end{align*}
$$


```R
exp(1.7)
```


5.4739473917272


So if the coefficient is 1.7, that implies that a 1-unit increase in the exposure will result in the mean ($\lambda$) increasing by a factor of roughly 5.5.

### 4.8 11M3 Explain why a logit link is appropriate for a binomial generalized linear model.

The binomial generalized linear model, the basic form is

$$
\begin{align*}
y &\sim \text{Binomial}(n, p) \\
\text{logit}(p) &= \alpha + \beta x \\
\end{align*}
$$

A logit link maps something which is a probability (so lives in the range [0, 1]) onto a linear model. In a binomial generalized linear model, the parameter $p$ in the binomial is generally what we are modelling; since this is a probability, the logit link is appropriate.

### 4.9 11M4 Explain why the log link is appropriate for a Poisson generalized linear model

The Poisson generalized linear model is

$$
\begin{align*}
y &\sim \text{Poisson}(\lambda) \\
\log \lambda &= \alpha + \beta x \\
\end{align*}
$$

The log link ensures that the parameter $\lambda$ will be positive (as it must be), and so is appropriate for this model.

### 4.10 11M5 What would it imply to use a logit link for the mean of a Poisson generalized linear mode? Can you think of a real research problem for which that would make sense?

In this case, the model would look like

$$
\begin{align*}
y &\sim \text{Poisson}(\lambda) \\
\text{logit} \lambda &= \alpha + \beta x \\
\end{align*}
$$

This would imply that the mean is itself a probability. That would mean that we are modelling a count where the mean value is a probability. This almost feels like a binomial regression - like if you flipped some coins a bunch of times and recorded the percent of heads as the 'count'?

### 4.11 11M6 State the contraints fro which the binomial and Poisson distributions have maximum entropy. Are the constrains different at all for binomial and Poisson? Why or why not?

The binomial and Poisson are both the maximum entropy distributions for a set of $n$ trials where the expected number of some kind of outcome is the same. Because the Poisson is just a special case of the binomial, the constraints are the same.

### 4.12 11M7 Use `quap` to construct a quadratic approximate posterior distribution for the chimpanzee model that includes a unique intercept for each actor, `m11.4`. Compare the quadratic approximation to the posterior distribution produced instead from MCMC. Can you explain both the differences and the similarities between the approximate and the MCMC distributions? Relax the prior on the actor intercepts to $\text{Normal}(0, 10)$. Re-estimate the posterior using both `ulam` and `quap`. Do the differences increase or decrease? Why?


```R
data(chimpanzees)
d <- chimpanzees
d$treatment <- 1 + d$prosoc_left + 2 * d$condition

# trimmed data list
dat_list <- list(
    pulled_left = d$pulled_left,
    actor = d$actor,
    treatment = as.integer(d$treatment)
)

m11.4_ulam <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + b[treatment],
        a[actor] ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = dat_list,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.4 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.5 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.5 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.4 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.4 seconds.
    Total execution time: 2.1 seconds.
    



```R
m11.4_quap <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + b[treatment],
        a[actor] ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = dat_list
)
```


```R
par(bg = 'white')
plot(precis(m11.4_ulam, depth = 2))
plot(precis(m11.4_quap, depth = 2))
```


    
![png](rethinking_ch11_output_148_0.png)
    



    
![png](rethinking_ch11_output_148_1.png)
    


These look basically identical. Let's try to plot them on the same set of axes to compare each paramter to the equivalent in the other model.


```R
par(bg = 'white')
plot(coeftab(m11.4_ulam, m11.4_quap))
```


    
![png](rethinking_ch11_output_150_0.png)
    


There is a very slight difference in the value for the second actor (one who only pulls right), but the rest seem to be the same. For that second actor, the quadratic approximation *underestimates* the value. This is probably due to asymmetry in the posterior distribution which is not captured by the (symmetric) normal.


```R
quap_samples <- extract.samples(m11.4_quap)
a2_quap <- quap_samples[['a']][, 2]
ulam_samples <- extract.samples(m11.4_ulam)
a2_ulam <- ulam_samples[['a']][, 2]

plot_df <- rbind(
    data.frame(
        x = a2_quap, type = "Quap"
    ),
    data.frame(
        x = a2_ulam, type = "Ulam"
    )
)
```


```R
ggplot(plot_df, aes(x, group = type, colour = type)) +
    geom_density(aes(y = after_stat(density)))
```


    
![png](rethinking_ch11_output_153_0.png)
    


We can see that `ulam` is putting slightly more weight on the upper part of the range, but not an enormous amount.

Now let's relax the actor priors to $\text{Normal}(0, 10)$ and see what happens.


```R
m11.4_ulam_relaxed <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + b[treatment],
        a[actor] ~ dnorm(0, 10),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = dat_list,
    chains = 4,
    log_lik = TRUE
)
m11.4_quap_relaxed <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + b[treatment],
        a[actor] ~ dnorm(0, 10),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = dat_list
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.5 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.6 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.5 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.6 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.6 seconds.
    Total execution time: 2.6 seconds.
    



```R
par(bg = 'white')
plot(coeftab(m11.4_ulam_relaxed, m11.4_quap_relaxed))
```


    
![png](rethinking_ch11_output_156_0.png)
    


This definitely looks like it accentuated the differences! 


```R
quap_samples <- extract.samples(m11.4_quap_relaxed)
a2_quap <- quap_samples[['a']][, 2]
ulam_samples <- extract.samples(m11.4_ulam_relaxed)
a2_ulam <- ulam_samples[['a']][, 2]

plot_df <- rbind(
    data.frame(
        x = a2_quap, type = "Quap"
    ),
    data.frame(
        x = a2_ulam, type = "Ulam"
    )
)
ggplot(plot_df, aes(x, group = type, colour = type)) +
    geom_density(aes(y = after_stat(density)))
```


    
![png](rethinking_ch11_output_158_0.png)
    


From this it is far more apparent that the asymmetry of the posterior is not being reflected in the normal approximation. Probably this is because with the weaker prior, the actual asymmetry is greater than with a strong prior.

### 4.13 11M8 Revisit the `data(Kline)` islands example. This time drop Hawaii from the sample and refit the models. What changes do you observe?


```R
data(Kline)
d <- Kline
d$P <- as.numeric(scale(log(d$population))) # this is required because otherwise P has a type of a 1-element list, which breaks STAN
# but for some reason didn't do that earlier
d$contact_id <- ifelse(d$contact == "high", 2, 1)

d_removed <- d[d$culture != 'Hawaii', ]
```


```R

dat <- list(
    T = d$total_tools,
    P = d$P,
    cid = d$contact_id
)
dat_removed <- list(
    T = d_removed$total_tools,
    P = d_removed$P,
    cid = d_removed$contact_id
)
m11.10_original <- ulam(
    alist(
        T ~ dpois(lambda),
        log(lambda) <- a[cid] + b[cid] * P,
        a[cid] ~ dnorm(3, 0.5),
        b[cid] ~ dnorm(0, 0.2)
    ),
    data = dat,
    chains = 4,
    log_lik = TRUE
)
m11.10_removed <- ulam(
    alist(
        T ~ dpois(lambda),
        log(lambda) <- a[cid] + b[cid] * P,
        a[cid] ~ dnorm(3, 0.5),
        b[cid] ~ dnorm(0, 0.2)
    ),
    data = dat_removed,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.
    
    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.6 seconds.
    



```R
####### ORIGINAL

# setting up the predictions
k <- PSIS(m11.10_original, pointwise = TRUE)$k
ns <- 100
P_seq <- seq(from = -5, to = 3, length.out = ns)
pop_seq <- exp(P_seq * 1.53 + 9) # 1.53 is the sd of log (population); 9 is the mean

#low contact (cid = 1) predictions
lambda <- link(m11.10_original, data = data.frame(P = P_seq, cid = 1))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
low_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 1
)

# high contact (cid = 2) predictions
lambda <- link(m11.10_original, data = data.frame(P = P_seq, cid = 2))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
high_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 2
)

predictions <- rbind(low_contact_df, high_contact_df)
predictions$cid <- factor(predictions$cid)
predictions$population <- pop_seq

plot_df <- data.frame(population = d$population, total_tools = d$total_tools, cid = factor(dat$cid), k = k, culture = d$culture)
ribbon_df <- data.frame(population = pop_seq, lower = predictions$lower, upper = predictions$upper, cid = factor(predictions$cid))
p <- ggplot() +
    geom_point(data = plot_df, mapping = aes(x = population, y = total_tools, shape = cid, size = 1 + normalize(k))) +
    geom_text(data = plot_df, mapping = aes(x = population, y = total_tools, label = culture), hjust = 0, vjust = 2) +
    geom_line(data = predictions, mapping = aes(population, mu, linetype = cid)) +
    geom_ribbon(data = ribbon_df, mapping = aes(x = population, ymin = lower, ymax = upper, group = cid), alpha = 0.2) + 
    coord_cartesian(xlim = c(0, 300000), ylim = c(0, 75)) +
    labs(title = "Original")
print(p)

#########
# With Hawaii removed

# setting up the predictions
k <- PSIS(m11.10_removed, pointwise = TRUE)$k
ns <- 100
P_seq <- seq(from = -5, to = 3, length.out = ns)
pop_seq <- exp(P_seq * 1.53 + 9) # 1.53 is the sd of log (population); 9 is the mean

#low contact (cid = 1) predictions
lambda <- link(m11.10_removed, data = data.frame(P = P_seq, cid = 1))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
low_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 1
)

# high contact (cid = 2) predictions
lambda <- link(m11.10_removed, data = data.frame(P = P_seq, cid = 2))
lmu <- apply(lambda, 2, mean)
lci <- apply(lambda, 2, PI)
high_contact_df <- data.frame(
    P = P_seq,
    mu = lmu,
    lower = lci[1, ],
    upper = lci[2, ],
    cid = 2
)

predictions <- rbind(low_contact_df, high_contact_df)
predictions$cid <- factor(predictions$cid)
predictions$population <- pop_seq

plot_df <- data.frame(population = d_removed$population, total_tools = d_removed$total_tools, cid = factor(dat_removed$cid), k = k, culture = d_removed$culture)
ribbon_df <- data.frame(population = pop_seq, lower = predictions$lower, upper = predictions$upper, cid = factor(predictions$cid))
p <- ggplot() +
    geom_point(data = plot_df, mapping = aes(x = population, y = total_tools, shape = cid, size = 1 + normalize(k))) +
    geom_text(data = plot_df, mapping = aes(x = population, y = total_tools, label = culture), hjust = 0, vjust = 2) +
    geom_line(data = predictions, mapping = aes(population, mu, linetype = cid)) +
    geom_ribbon(data = ribbon_df, mapping = aes(x = population, ymin = lower, ymax = upper, group = cid), alpha = 0.2) + 
    coord_cartesian(xlim = c(0, 300000), ylim = c(0, 75)) +
    labs(title = "Hawaii removed")
print(p)
```

    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    
    Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
    



    
![png](rethinking_ch11_output_163_1.png)
    



    
![png](rethinking_ch11_output_163_2.png)
    


Hawaii is obviously a very influential point. With it removed, the predictions for all contact types (high and low) are lower than they otherwise would be. In addition, the unexpected behaviour where the high-contact line dips below the low-contact line no longer happens; now the high-contact line is consistently above the low-contact one, as we expect.

### 4.14 11H1 Use WAIC or PSIS to compare the chimpanzee model that includes a unique intercept for each actor, m11.4, to the simpler models fit in the same section. Interpret the results.


```R
data(chimpanzees)
d <- chimpanzees
d$treatment <- 1 + d$prosoc_left + 2 * d$condition

# actually the revised version
m11.1 <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a,
        a ~ dnorm(0, 1.5)
    ),
    data = d
)

m11.2 <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a + b[treatment],
        a ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 10)
    ),
    data = d
)

m11.3 <- quap(
    alist(
        pulled_left <- dbinom(1, p),
        logit(p) <- a + b[treatment],
        a ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = d
)

# trimmed data list
dat_list <- list(
    pulled_left = d$pulled_left,
    actor = d$actor,
    treatment = as.integer(d$treatment)
)

m11.4 <- ulam(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a[actor] + b[treatment],
        a[actor] ~ dnorm(0, 1.5),
        b[treatment] ~ dnorm(0, 0.5)
    ),
    data = dat_list,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.5 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.4 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.4 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.4 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.4 seconds.
    Total execution time: 2.2 seconds.
    



```R
comparison <- compare(m11.1, m11.2, m11.3, m11.4)
comparison
```

    Warning message in compare(m11.1, m11.2, m11.3, m11.4):
    “Not all model fits of same class.
    This is usually a bad idea, because it implies they were fit by different algorithms.
    Check yourself, before you wreck yourself.”



<table class="dataframe">
<caption>A compareIC: 4 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>m11.4</th><td>532.1620</td><td>18.962754</td><td>  0.0000</td><td>      NA</td><td>8.5660325</td><td>1.000000e+00</td></tr>
	<tr><th scope=row>m11.3</th><td>682.4367</td><td> 9.045846</td><td>150.2748</td><td>18.40466</td><td>3.6044256</td><td>2.334814e-33</td></tr>
	<tr><th scope=row>m11.2</th><td>683.0660</td><td> 9.654009</td><td>150.9040</td><td>18.48263</td><td>4.0061776</td><td>1.704546e-33</td></tr>
	<tr><th scope=row>m11.1</th><td>687.9404</td><td> 7.176299</td><td>155.7785</td><td>18.98764</td><td>0.9998617</td><td>1.489826e-34</td></tr>
</tbody>
</table>



Looking at these, the model with the actor-specific intercept is clearly better (about 6-7 SE above the next one). All of the rest are basically in the same ballpark, but again are significantly lower than the actor-specific model.


```R
comparison_df <- as.data.frame(comparison)
comparison_df <- cbind(model = rownames(comparison_df), comparison_df)
rownames(comparison_df) <- 1:nrow(comparison_df)
comparison_df
```


<table class="dataframe">
<caption>A data.frame: 4 × 7</caption>
<thead>
	<tr><th></th><th scope=col>model</th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>m11.4</td><td>532.1620</td><td>18.962754</td><td>  0.0000</td><td>      NA</td><td>8.5660325</td><td>1.000000e+00</td></tr>
	<tr><th scope=row>2</th><td>m11.3</td><td>682.4367</td><td> 9.045846</td><td>150.2748</td><td>18.40466</td><td>3.6044256</td><td>2.334814e-33</td></tr>
	<tr><th scope=row>3</th><td>m11.2</td><td>683.0660</td><td> 9.654009</td><td>150.9040</td><td>18.48263</td><td>4.0061776</td><td>1.704546e-33</td></tr>
	<tr><th scope=row>4</th><td>m11.1</td><td>687.9404</td><td> 7.176299</td><td>155.7785</td><td>18.98764</td><td>0.9998617</td><td>1.489826e-34</td></tr>
</tbody>
</table>




```R
ggplot(data.frame(
    model = comparison_df$model,
    WAIC = comparison_df$WAIC,
    lower = comparison_df$WAIC - comparison_df$SE,
    upper = comparison_df$WAIC + comparison_df$SE
), aes(WAIC, model)) +
    geom_point() +
    geom_linerange(aes(xmin = lower, xmax = upper))
```


    
![png](rethinking_ch11_output_170_0.png)
    


Looking at the plot makes it even more clear - m11.4 is clearly much better than the other ones.

### 4.15 11H2 The data contained in `library(MASS); data(eagles)` are records of slamon pirating attempts by Bald Eagles in Washington State. See `?eagles` for more information. WHile one eagle feeds, sometimes another will swoop in and try to steal the salmon from it. Call the feeding eagle the "victim" and the thief the "pirate". Use the available data to build a binomial GLM of successful pirating attempts.
#### 4.15.1 a) Consider the following model:
$$
\begin{align*}
y_i &\sim \text{Binomial}(n_i, p_i) \\
\text{logit}(p_i) &= \alpha + \beta_P P_i + \beta_V V_i + \beta_A A_i \\
\alpha &\sim \text{Normal}(0, 1.5) \\
\beta_P, \beta_V, \beta_A &\sim \text{Normal}(0, 0.5)
\end{align*}
$$
### 4.16 where $y$ is the number of successful attempts, $n$ is the total number of attempts, $P$ is a dunmmy variable indicating whether or not the pirate had large body size, $V$ is a dummy variable indicating whether or not the victim had a large body size, and $A$ is a dummy variable for whether the pirate was an adult. Fit the model above to the `eagles` data, using both `quap` and `ulam`. Is the quadratic approximation OK?


```R
library(MASS)
data(eagles)
d <- eagles
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 5</caption>
<thead>
	<tr><th></th><th scope=col>y</th><th scope=col>n</th><th scope=col>P</th><th scope=col>A</th><th scope=col>V</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>17</td><td>24</td><td>L</td><td>A</td><td>L</td></tr>
	<tr><th scope=row>2</th><td>29</td><td>29</td><td>L</td><td>A</td><td>S</td></tr>
	<tr><th scope=row>3</th><td>17</td><td>27</td><td>L</td><td>I</td><td>L</td></tr>
	<tr><th scope=row>4</th><td>20</td><td>20</td><td>L</td><td>I</td><td>S</td></tr>
	<tr><th scope=row>5</th><td> 1</td><td>12</td><td>S</td><td>A</td><td>L</td></tr>
	<tr><th scope=row>6</th><td>15</td><td>16</td><td>S</td><td>A</td><td>S</td></tr>
</tbody>
</table>




```R
# modify the columns to be 0 or 1
d$P_actual <- ifelse(d$P == "L", 1, 0)
d$A_actual <- ifelse(d$A == "A", 1, 0)
d$V_actual <- ifelse(d$V == "L", 1, 0)
```


```R
eagles_model_quap <- quap(
    alist(
        y ~ dbinom(n, p),
        logit(p) <- a + b_p * P_actual + b_v * V_actual * b_a * A_actual,
        a ~ dnorm(0, 1.5),
        c(b_p, b_v, b_a) ~ dnorm(0, 0.5)
    ),
    data = d
)
```


```R
eagles_data <- list(
    y = d$y,
    n = d$n,
    P_actual = d$P_actual,
    V_actual = d$V_actual,
    A_actual = d$A_actual
)
eagles_model_ulam <- ulam(
    alist(
        y ~ dbinom(n, p),
        logit(p) <- a + b_p * P_actual + b_v * V_actual * b_a * A_actual,
        a ~ dnorm(0, 1.5),
        c(b_p, b_v, b_a) ~ dnorm(0, 0.5)
    ),
    data = eagles_data,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.
    



```R
precis(eagles_model_quap)
```


<table class="dataframe">
<caption>A precis: 4 × 4</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>-0.3910487</td><td>0.2499844</td><td>-0.7905721</td><td>0.0084747</td></tr>
	<tr><th scope=row>b_p</th><td> 1.6208560</td><td>0.2882221</td><td> 1.1602214</td><td>2.0814907</td></tr>
	<tr><th scope=row>b_v</th><td> 0.4414322</td><td>0.5363755</td><td>-0.4157994</td><td>1.2986638</td></tr>
	<tr><th scope=row>b_a</th><td>-0.4414347</td><td>0.5363762</td><td>-1.2986674</td><td>0.4157980</td></tr>
</tbody>
</table>




```R
precis(eagles_model_ulam)
```


<table class="dataframe">
<caption>A precis: 4 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>-0.385271355</td><td>0.2364014</td><td>-0.7550741</td><td>-0.01733659</td><td>1.001085</td><td> 973.5718</td></tr>
	<tr><th scope=row>b_a</th><td> 0.009930345</td><td>0.5783893</td><td>-0.8884777</td><td> 0.92075510</td><td>1.003419</td><td> 785.5355</td></tr>
	<tr><th scope=row>b_v</th><td>-0.012529652</td><td>0.5924564</td><td>-0.9644759</td><td> 0.87919503</td><td>1.002272</td><td> 808.4992</td></tr>
	<tr><th scope=row>b_p</th><td> 1.631829008</td><td>0.2884735</td><td> 1.1797627</td><td> 2.08443205</td><td>1.004493</td><td>1007.7628</td></tr>
</tbody>
</table>




```R
quap_df <- as.data.frame(precis(eagles_model_quap))
quap_df <- cbind(variable = rownames(quap_df), quap_df)
rownames(quap_df) <- 1:nrow(quap_df)
quap_df$model <- "Quap"
ulam_df <- as.data.frame(precis(eagles_model_ulam))
ulam_df <- cbind(variable = rownames(ulam_df), ulam_df)
rownames(ulam_df) <- 1:nrow(ulam_df)
ulam_df$model <- "Ulam"
quap_df
ulam_df
```


<table class="dataframe">
<caption>A data.frame: 4 × 6</caption>
<thead>
	<tr><th></th><th scope=col>variable</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>model</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>a  </td><td>-0.3910487</td><td>0.2499844</td><td>-0.7905721</td><td>0.0084747</td><td>Quap</td></tr>
	<tr><th scope=row>2</th><td>b_p</td><td> 1.6208560</td><td>0.2882221</td><td> 1.1602214</td><td>2.0814907</td><td>Quap</td></tr>
	<tr><th scope=row>3</th><td>b_v</td><td> 0.4414322</td><td>0.5363755</td><td>-0.4157994</td><td>1.2986638</td><td>Quap</td></tr>
	<tr><th scope=row>4</th><td>b_a</td><td>-0.4414347</td><td>0.5363762</td><td>-1.2986674</td><td>0.4157980</td><td>Quap</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A data.frame: 4 × 8</caption>
<thead>
	<tr><th></th><th scope=col>variable</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th><th scope=col>model</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>a  </td><td>-0.385271355</td><td>0.2364014</td><td>-0.7550741</td><td>-0.01733659</td><td>1.001085</td><td> 973.5718</td><td>Ulam</td></tr>
	<tr><th scope=row>2</th><td>b_a</td><td> 0.009930345</td><td>0.5783893</td><td>-0.8884777</td><td> 0.92075510</td><td>1.003419</td><td> 785.5355</td><td>Ulam</td></tr>
	<tr><th scope=row>3</th><td>b_v</td><td>-0.012529652</td><td>0.5924564</td><td>-0.9644759</td><td> 0.87919503</td><td>1.002272</td><td> 808.4992</td><td>Ulam</td></tr>
	<tr><th scope=row>4</th><td>b_p</td><td> 1.631829008</td><td>0.2884735</td><td> 1.1797627</td><td> 2.08443205</td><td>1.004493</td><td>1007.7628</td><td>Ulam</td></tr>
</tbody>
</table>




```R
combined_df <- rbind(data.frame(
    variable = quap_df$variable,
    model = quap_df$model,
    mean = quap_df$mean,
    sd = quap_df$sd
), data.frame(
    variable = ulam_df$variable,
    model = ulam_df$model,
    mean = ulam_df$mean,
    sd = ulam_df$sd
))
combined_df$lower <- combined_df$mean - combined_df$sd
combined_df$upper <- combined_df$mean + combined_df$sd
combined_df
```


<table class="dataframe">
<caption>A data.frame: 8 × 6</caption>
<thead>
	<tr><th scope=col>variable</th><th scope=col>model</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>lower</th><th scope=col>upper</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>a  </td><td>Quap</td><td>-0.391048699</td><td>0.2499844</td><td>-0.6410331</td><td>-0.14106427</td></tr>
	<tr><td>b_p</td><td>Quap</td><td> 1.620856033</td><td>0.2882221</td><td> 1.3326339</td><td> 1.90907817</td></tr>
	<tr><td>b_v</td><td>Quap</td><td> 0.441432184</td><td>0.5363755</td><td>-0.0949433</td><td> 0.97780766</td></tr>
	<tr><td>b_a</td><td>Quap</td><td>-0.441434735</td><td>0.5363762</td><td>-0.9778109</td><td> 0.09494142</td></tr>
	<tr><td>a  </td><td>Ulam</td><td>-0.385271355</td><td>0.2364014</td><td>-0.6216728</td><td>-0.14886991</td></tr>
	<tr><td>b_a</td><td>Ulam</td><td> 0.009930345</td><td>0.5783893</td><td>-0.5684590</td><td> 0.58831964</td></tr>
	<tr><td>b_v</td><td>Ulam</td><td>-0.012529652</td><td>0.5924564</td><td>-0.6049860</td><td> 0.57992674</td></tr>
	<tr><td>b_p</td><td>Ulam</td><td> 1.631829008</td><td>0.2884735</td><td> 1.3433555</td><td> 1.92030255</td></tr>
</tbody>
</table>




```R
ggplot(combined_df, aes(mean, variable, group = interaction(variable, model), colour = model)) +
    geom_point(position=ggstance::position_dodgev(height=0.3)) +
    geom_linerange(aes(xmin = lower, xmax = upper), position=ggstance::position_dodgev(height=0.3))
```


    
![png](rethinking_ch11_output_181_0.png)
    


From this, it seems clear that the quadratic approximation is poor for $\beta_A$ and $\beta_V$, but good for $\beta_P$ and $\alpha$.

#### 4.16.1 b) Now interpret the estimates. If the quadratic approximation turned out okay, then it's okay to use the `quap` estimates. Otherwise stick to `ulam` estimates. The plot the posterior predictions. Compute and display both (1) the predicted **probability** of success and its 89% interval for each row ($i$) in the data, as well as (2) the predicted success **count** and its 89% interval. What different information does each type of posterior prediction provide?


```R
samples <- extract.samples(eagles_model_ulam)
lapply(samples, head)
```


<dl>
	<dt>$a</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 1 of type dbl</caption>
<tbody>
	<tr><td>-0.818957</td></tr>
	<tr><td>-0.753176</td></tr>
	<tr><td>-0.597761</td></tr>
	<tr><td>-0.325199</td></tr>
	<tr><td>-0.437738</td></tr>
	<tr><td>-0.332195</td></tr>
</tbody>
</table>
</dd>
	<dt>$b_a</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 1 of type dbl</caption>
<tbody>
	<tr><td> 0.9494830</td></tr>
	<tr><td> 0.8638590</td></tr>
	<tr><td> 0.3309760</td></tr>
	<tr><td>-0.0180856</td></tr>
	<tr><td>-0.3443510</td></tr>
	<tr><td>-0.3799370</td></tr>
</tbody>
</table>
</dd>
	<dt>$b_v</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 1 of type dbl</caption>
<tbody>
	<tr><td> 0.139395000</td></tr>
	<tr><td>-0.000150099</td></tr>
	<tr><td>-0.327047000</td></tr>
	<tr><td> 0.307270000</td></tr>
	<tr><td> 0.132654000</td></tr>
	<tr><td> 0.261492000</td></tr>
</tbody>
</table>
</dd>
	<dt>$b_p</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 1 of type dbl</caption>
<tbody>
	<tr><td>1.74057</td></tr>
	<tr><td>1.73009</td></tr>
	<tr><td>1.88860</td></tr>
	<tr><td>1.33775</td></tr>
	<tr><td>1.77496</td></tr>
	<tr><td>1.62158</td></tr>
</tbody>
</table>
</dd>
	<dt>$p</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 8 of type dbl</caption>
<tbody>
	<tr><td>0.741536</td><td>0.715370</td><td>0.715370</td><td>0.715370</td><td>0.334789</td><td>0.305985</td><td>0.305985</td><td>0.305985</td></tr>
	<tr><td>0.726470</td><td>0.726495</td><td>0.726495</td><td>0.726495</td><td>0.320102</td><td>0.320130</td><td>0.320130</td><td>0.320130</td></tr>
	<tr><td>0.765414</td><td>0.784289</td><td>0.784289</td><td>0.784289</td><td>0.330482</td><td>0.354856</td><td>0.354856</td><td>0.354856</td></tr>
	<tr><td>0.732432</td><td>0.733519</td><td>0.733519</td><td>0.733519</td><td>0.418057</td><td>0.419409</td><td>0.419409</td><td>0.419409</td></tr>
	<tr><td>0.784407</td><td>0.792032</td><td>0.792032</td><td>0.792032</td><td>0.381445</td><td>0.392280</td><td>0.392280</td><td>0.392280</td></tr>
	<tr><td>0.766747</td><td>0.784043</td><td>0.784043</td><td>0.784043</td><td>0.393757</td><td>0.417707</td><td>0.417707</td><td>0.417707</td></tr>
</tbody>
</table>
</dd>
</dl>




```R
df <- data.frame(
    P = numeric(),
    V = numeric(),
    A = numeric(),
    logit_p = numeric()
)
for (P in c(0, 1)) {
    for (V in c(0, 1)) {
        for (A in c(0, 1)) {
            for (i in 1:length(samples[['a']])) {
                df <- rbind(
                    df,
                    data.frame(
                        P = P,
                        V = V,
                        A = A,
                        logit_p = samples[['a']][i] + samples[['b_p']][i] * P + samples[['b_v']][i] * V + samples[['b_a']][i] * A
                    )
                )
            }
        }
    }
}
df$p <- inv_logit(df$logit_p)
```


```R
ggplot(df, aes(p, group = interaction(P, V, A), colour = interaction(P, V, A))) +
    geom_density(aes(y = after_stat(density)))
```


    
![png](rethinking_ch11_output_186_0.png)
    


Now for the counts!


```R
# CALCULATION FOR N IS NOT CORRECT - ALWAYS RETURNING 24
df$count <- rep(NA, nrow(df))
df$n <- rep(NA, nrow(df))
for (i in 1:(nrow(df))) {
    row <- df[i, ]
    P <- row[, 'P'][1]
    V <- row[, 'V'][1]
    A <- row[, 'A'][1]
    matching <- d[d$P_actual == P & d$V_actual == V & d$A_actual == A, ]
    n <- matching[, 'n'][1]
    p <- df[i, 'p']
    df[i, 'n'] <- n
    df[i, 'count'] <- rbinom(1, n, p)
}
```


```R
ggplot(df, aes(count, group = interaction(P, V, A), colour = interaction(P, V, A))) +
    geom_density(aes(y = after_stat(density)))
```


    
![png](rethinking_ch11_output_189_0.png)
    


This does provide slightly different information. While generally the same patterns hold, there are differences due to the varying number of attempts made by the different categories.

#### 4.16.2 c) Now try to improve the model. Consider an interaction between the pirate's size and age. Compare this model to the previous one using WAIC. Interpret the results.


```R
d$I_actual <- d$P_actual * d$A_actual

eagles_data <- list(
    y = d$y,
    n = d$n,
    P_actual = d$P_actual,
    V_actual = d$V_actual,
    A_actual = d$A_actual,
    I_actual = d$I_actual
)

eagles_model_ulam_interaction <- ulam(
    alist(
        y ~ dbinom(n, p),
        logit(p) <- a + b_p * P_actual + b_v * V_actual * b_a * A_actual + b_i * I_actual,
        a ~ dnorm(0, 1.5),
        c(b_p, b_v, b_a, b_i) ~ dnorm(0, 0.5)
    ),
    data = eagles_data,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.5 seconds.
    



```R
par(bg = 'white')
plot(precis(eagles_model_ulam_interaction))
plot(precis(eagles_model_ulam))
```


    
![png](rethinking_ch11_output_192_0.png)
    



    
![png](rethinking_ch11_output_192_1.png)
    



```R
compare(eagles_model_ulam_interaction, eagles_model_ulam, func = WAIC)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>eagles_model_ulam_interaction</th><td>116.7276</td><td>34.34376</td><td>0.0000000</td><td>     NA</td><td>18.89529</td><td>0.5668883</td></tr>
	<tr><th scope=row>eagles_model_ulam</th><td>117.2659</td><td>33.24724</td><td>0.5383333</td><td>6.88195</td><td>16.52250</td><td>0.4331117</td></tr>
</tbody>
</table>



So here the interaction coefficient $b_I$ seems to have an effect that we care about, but looking at the models they are virtually indisinguishable. This is not surprising; since the age and size variables are binary, their 'interaction' basically just tests if the pirate is larger *and* older, and since the effect of being larger ($P$) is already so large, this information is essentially just encoded there.

### 4.17 11H3 The data contained in `data(salamanders)` are counts of salamanders (*Plethodon elongatus*) from 47 different 59m$^2$ plots in northern California. The column `SALAMAN` is the count in each plot, and the columns `PCTCOVER` and `FORESTAGE` are percent of ground cover and age of trees in the plot, respectively. You will model `SALAMA` as a Poisson variable.
#### 4.17.1 a) Model the relationship between density and percent cover, using a log link (same as the examples in the book and lexture). Use weakly informative proiors of your choosing. Checl the quadratic approximation again, by comparing `quap` to `ulam`. Then plot the expected counts and their 89% intervals against percent cover. In what ways does the model do a good job? A bad job?


```R
data(salamanders)
d <- salamanders
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>SITE</th><th scope=col>SALAMAN</th><th scope=col>PCTCOVER</th><th scope=col>FORESTAGE</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>13</td><td>85</td><td>316</td></tr>
	<tr><th scope=row>2</th><td>2</td><td>11</td><td>86</td><td> 88</td></tr>
	<tr><th scope=row>3</th><td>3</td><td>11</td><td>90</td><td>548</td></tr>
	<tr><th scope=row>4</th><td>4</td><td> 9</td><td>88</td><td> 64</td></tr>
	<tr><th scope=row>5</th><td>5</td><td> 8</td><td>89</td><td> 43</td></tr>
	<tr><th scope=row>6</th><td>6</td><td> 7</td><td>83</td><td>368</td></tr>
</tbody>
</table>




```R
# first let's get an idea for the relationship here
ggplot(d, aes(PCTCOVER, SALAMAN)) +
    geom_point()
```


    
![png](rethinking_ch11_output_197_0.png)
    



```R
# coming up with some priors
# model is 
# a ~ dpois(lambda)
# lambda <- a + b_p * P
# a ~ dnorm(???)
# b_p ~ dnorm(???)
# NB I got these priors mostly by just random guessing and fiddling about until things looked reasonable

N <- 100
a <- rnorm(N, 0, 1)
b_P <- rnorm(N, 0, 0.005)
P <- seq(0, 100, by = 1)
plot_df <- data.frame(P = numeric(), lambda = numeric(), i = numeric())
for (i in 1:N) {
    lambda <- exp(a[i] + b_P[i] * P)
    plot_df <- rbind(
        plot_df,
        data.frame(
            P = P,
            lambda = lambda,
            i = i
        )
    )
}

ggplot(d, aes(PCTCOVER, SALAMAN)) +
    geom_point() +
    geom_line(data = plot_df, aes(P, lambda, group = i), alpha = 0.2)
```


    
![png](rethinking_ch11_output_198_0.png)
    



```R
salamander_model <- alist(
    s ~ dpois(lambda),
    log(lambda) <- a + b_P * P,
    a ~ dnorm(0, 1),
    b_P ~ dnorm(0, 0.005)
)

data <- list(
    s = d$SALAMAN,
    P = d$PCTCOVER
)

salamander_model_quap <- quap(salamander_model, data = data)
salamander_model_ulam <- ulam(salamander_model, data = data, chains = 4, log_lik = TRUE)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.1 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.1 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.1 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.5 seconds.
    


    Warning: 1 of 4 chains had an E-BFMI less than 0.2.
    See https://mc-stan.org/misc/warnings for details.
    
    



```R
par(bg = 'white')
plot(precis(salamander_model_quap))
plot(precis(salamander_model_ulam))
```


    
![png](rethinking_ch11_output_200_0.png)
    



    
![png](rethinking_ch11_output_200_1.png)
    



```R
salamander_quap_samples <- extract.samples(salamander_model_quap)
salamander_ulam_samples <- extract.samples(salamander_model_ulam)
```


```R
alpha <- 0.11
a_quap <- salamander_quap_samples[['a']]
a_quap_mean <- mean(a_quap)
a_quap_bounds <- quantile(a_quap, c(alpha / 2, 1 - alpha / 2))
a_ulam <- salamander_ulam_samples[['a']]
a_ulam_mean <- mean(a_ulam)
a_ulam_bounds <- quantile(a_ulam, c(alpha / 2, 1 - alpha / 2))

b_P_quap <- salamander_quap_samples[['b_P']]
b_P_ulam <- salamander_ulam_samples[['b_P']]
b_P_quap_mean <- mean(b_P_quap)
b_P_quap_bounds <- quantile(b_P_quap, c(alpha / 2, 1 - alpha / 2))
b_P_ulam_mean <- mean(b_P_ulam)
b_P_ulam_bounds <- quantile(b_P_ulam, c(alpha / 2, 1 - alpha / 2))

# NB This is a dumb way to do it - probably the better way would be to take the standard error and calculate the range from that
# or basically any other way
comparison_df <- data.frame(
    variable = rep(c("a", "b_P"), each = 2),
    model = rep(c("Quap", "Ulam"), 2),
    mean = c(a_quap_mean, a_ulam_mean, b_P_quap_mean, b_P_ulam_mean),
    lower = c(a_quap_bounds[1], a_ulam_bounds[1], b_P_quap_bounds[1], b_P_ulam_bounds[2]),
    upper = c(a_quap_bounds[2], a_ulam_bounds[2], b_P_quap_bounds[1], b_P_ulam_bounds[2])
)
comparison_df
```


<table class="dataframe">
<caption>A data.frame: 4 × 5</caption>
<thead>
	<tr><th scope=col>variable</th><th scope=col>model</th><th scope=col>mean</th><th scope=col>lower</th><th scope=col>upper</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>a  </td><td>Quap</td><td>-0.33347394</td><td>-0.70510915</td><td>0.04411952</td></tr>
	<tr><td>a  </td><td>Ulam</td><td>-0.22044404</td><td>-0.63281211</td><td>0.08046080</td></tr>
	<tr><td>b_P</td><td>Quap</td><td> 0.01807941</td><td> 0.01352625</td><td>0.01352625</td></tr>
	<tr><td>b_P</td><td>Ulam</td><td> 0.07445065</td><td> 0.25635706</td><td>0.25635706</td></tr>
</tbody>
</table>




```R
ggplot(comparison_df, aes(group = interaction(variable, model), colour = model)) +
    geom_pointrange(aes(x = mean, y = variable, xmin = lower, xmax = upper), position = ggstance::position_dodgev(height = 0.3))
```


    
![png](rethinking_ch11_output_203_0.png)
    


Basically these models give the same results - not a huge difference between them.

Now let's take a look at the posterior predictions!


```R
cover_percent <- seq(0, 100, length.out = 100)
prediction_data <- data.frame(P = cover_percent)
# why is this getting lambda? What is the link function doing here? How does it know to get this instead of s, log(p), &c.?
lambda <- link(salamander_model_ulam, data = prediction_data)
```


```R
lambda_mean <- apply(lambda, 2, mean)
lambda_bounds <- apply(lambda, 2, function(x) PI(x, 1 - alpha))
plot_df <- data.frame(P = cover_percent, mean = lambda_mean, lower = lambda_bounds[1, ], upper = lambda_bounds[2, ])
```


```R
ggplot() +
    geom_point(data = d, aes(PCTCOVER, SALAMAN)) +
    geom_line(data = plot_df, mapping = aes(x = P, y = mean)) +
    geom_ribbon(data = plot_df, mapping = aes(x = P, ymin = lower, ymax = upper), alpha = 0.2)
```


    
![png](rethinking_ch11_output_207_0.png)
    


This model is clearly doing a pretty poor job of fitting to the data. It's slightly better at the lower end, although it consistently predicts too high. However, it has real trouble at the 75% - 100% forest cover range, where the variation is enormous. It seems pretty clear that we are missing some important piece of data here.


```R
# getting the priors
N <- 100
a <- rnorm(N, 0, 1)
b_A <- rnorm(N, 0, 0.005)
A <- seq(0, 700, by = 1)
plot_df <- data.frame(A = numeric(), lambda = numeric(), i = numeric())
for (i in 1:N) {
    lambda <- exp(a[i] + b_A[i] * A)
    plot_df <- rbind(
        plot_df,
        data.frame(
            A = A,
            lambda = lambda,
            i = i
        )
    )
}

ggplot(d, aes(FORESTAGE, SALAMAN)) +
    geom_point() +
    geom_line(data = plot_df, aes(A, lambda, group = i), alpha = 0.2) +
    coord_cartesian(ylim = c(0, 100))
```


    
![png](rethinking_ch11_output_209_0.png)
    



```R
new_salamander_model <- alist(
    s ~ dpois(lambda),
    log(lambda) <- a + b_P * P + b_A * A,
    a ~ dnorm(0, 1),
    b_P ~ dnorm(0, 0.005),
    b_A ~ dnorm(0, 0.005)
)

data <- list(
    s = d$SALAMAN,
    P = d$PCTCOVER,
    A = d$FORESTAGE
)

forest_age_salamander_model <- ulam(new_salamander_model, data = data, chains = 4, log_lik = TRUE)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.2 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.1 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.4 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.1 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.2 seconds.
    Total execution time: 1.1 seconds.
    


    Warning: 1 of 2000 (0.0%) transitions hit the maximum treedepth limit of 10.
    See https://mc-stan.org/misc/warnings for details.
    
    
    Warning: 1 of 4 chains have a NaN E-BFMI.
    See https://mc-stan.org/misc/warnings for details.
    
    



```R
par(bg = 'white')
plot(precis(forest_age_salamander_model))
```


    
![png](rethinking_ch11_output_211_0.png)
    


Hmmm, so the model seems pretty sure that forest age doesn't have anything to do with the number of salamanders. Let's compare the models.


```R
compare(forest_age_salamander_model, salamander_model_ulam)
```


    Error in `[.data.frame`(result, , result_order): undefined columns selected
    Traceback:


    1. compare(forest_age_salamander_model, salamander_model_ulam)

    2. result[, result_order]

    3. `[.data.frame`(result, , result_order)

    4. stop("undefined columns selected")


From this, it seems like this model is doing a worse job of predicting the number of salamanders.


```R
ggplot(d, aes(PCTCOVER, FORESTAGE)) +
    geom_point() +
    geom_smooth(method = 'lm')
```

    [1m[22m`geom_smooth()` using formula = 'y ~ x'



    
![png](rethinking_ch11_output_215_1.png)
    


Hmmmm... it doesn't look like there is a strong relationship between the cover and the age. However, since there is some relationship it looks like there might be some multicollinearity occurring (since there's a weak correlation between the forest age and cover).

### 4.18 11H4 The data in `data(NWOGrants)` are outcomes for scientific funding applications for the Netherlands Organization for Scientific Research (NWO) for 2010 - 2012. These data have a very similar structure to the `UCBAdmit` data discussed in the chapter. I want you to consider a similar question: What are the total and indirect causal effects of gender on grant awards? Consider a mediation path (a pipe) through `discipline`. Draw the corresponding DAG and then use one or more binomial GLMs to answer the question. What is your causal interpretation? If NWO's goal is to equalize rates of funding between men and women, what type of intervention would be most effective?


```R
data(NWOGrants)
d <- NWOGrants
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>discipline</th><th scope=col>gender</th><th scope=col>applications</th><th scope=col>awards</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Chemical sciences</td><td>m</td><td> 83</td><td>22</td></tr>
	<tr><th scope=row>2</th><td>Chemical sciences</td><td>f</td><td> 39</td><td>10</td></tr>
	<tr><th scope=row>3</th><td>Physical sciences</td><td>m</td><td>135</td><td>26</td></tr>
	<tr><th scope=row>4</th><td>Physical sciences</td><td>f</td><td> 39</td><td> 9</td></tr>
	<tr><th scope=row>5</th><td>Physics          </td><td>m</td><td> 67</td><td>18</td></tr>
	<tr><th scope=row>6</th><td>Physics          </td><td>f</td><td>  9</td><td> 2</td></tr>
</tbody>
</table>




```R
summary(NWOGrants)
```


                   discipline gender  applications        awards     
     Chemical sciences  :2    f:9    Min.   :  9.00   Min.   : 2.00  
     Earth/life sciences:2    m:9    1st Qu.: 69.75   1st Qu.:14.00  
     Humanities         :2           Median :130.50   Median :24.00  
     Interdisciplinary  :2           Mean   :156.83   Mean   :25.94  
     Medical sciences   :2           3rd Qu.:219.75   3rd Qu.:32.75  
     Physical sciences  :2           Max.   :425.00   Max.   :65.00  
     (Other)            :6                                           



```R
d$ratio <- d$awards / d$applications
ggplot(d, aes(discipline, ratio, fill = gender)) +
    geom_bar(stat = 'identity', position = position_dodge())
```


    
![png](rethinking_ch11_output_220_0.png)
    



```R
plot_df <- data.frame(discipline = unique(d$discipline))
plot_df$difference <- NA
for (i in 1:nrow(plot_df)) {
    relevant <- d[d$discipline == plot_df[i, 'discipline'], ]
    plot_df[i, 'difference'] <- relevant[relevant$gender == 'm', 'ratio'][1] - relevant[relevant$gender == 'f', 'ratio'][1]
}
plot_df
```


<table class="dataframe">
<caption>A data.frame: 9 × 2</caption>
<thead>
	<tr><th scope=col>discipline</th><th scope=col>difference</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Chemical sciences  </td><td> 0.008649985</td></tr>
	<tr><td>Physical sciences  </td><td>-0.038176638</td></tr>
	<tr><td>Physics            </td><td> 0.046434494</td></tr>
	<tr><td>Humanities         </td><td>-0.049292823</td></tr>
	<tr><td>Technical sciences </td><td>-0.050947261</td></tr>
	<tr><td>Interdisciplinary  </td><td>-0.103663004</td></tr>
	<tr><td>Earth/life sciences</td><td> 0.100732601</td></tr>
	<tr><td>Social sciences    </td><td> 0.038026751</td></tr>
	<tr><td>Medical sciences   </td><td> 0.076216641</td></tr>
</tbody>
</table>




```R
ggplot(plot_df, aes(discipline, difference, fill = difference < 0)) +
    geom_bar(stat = 'identity')
```


    
![png](rethinking_ch11_output_222_0.png)
    



```R
par(bg = 'white')
dag <- dagitty('dag{G -> D -> A; G -> A}')
drawdag(dag)
```


    
![png](rethinking_ch11_output_223_0.png)
    



```R
# total effect: A ~ G
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 5</caption>
<thead>
	<tr><th></th><th scope=col>discipline</th><th scope=col>gender</th><th scope=col>applications</th><th scope=col>awards</th><th scope=col>ratio</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Chemical sciences</td><td>m</td><td> 83</td><td>22</td><td>0.2650602</td></tr>
	<tr><th scope=row>2</th><td>Chemical sciences</td><td>f</td><td> 39</td><td>10</td><td>0.2564103</td></tr>
	<tr><th scope=row>3</th><td>Physical sciences</td><td>m</td><td>135</td><td>26</td><td>0.1925926</td></tr>
	<tr><th scope=row>4</th><td>Physical sciences</td><td>f</td><td> 39</td><td> 9</td><td>0.2307692</td></tr>
	<tr><th scope=row>5</th><td>Physics          </td><td>m</td><td> 67</td><td>18</td><td>0.2686567</td></tr>
	<tr><th scope=row>6</th><td>Physics          </td><td>f</td><td>  9</td><td> 2</td><td>0.2222222</td></tr>
</tbody>
</table>




```R
nwo_data <- list(
    applications = d$applications,
    awards = d$awards,
    gid = ifelse(d$gender == "m", 1, 2),
    did = rep(1:length(unique(d$discipline)), each = 2) # department index
)
total_effects_model <- ulam(
    alist(
        awards ~ dbinom(applications, p),
        logit(p) <- a[gid],
        a[gid] ~ dnorm(0, 1.5)
    ),
    data = nwo_data,
    chains = 4,
    log_lik = TRUE
)

indirect_effects_model <- ulam(
    alist(
        awards ~ dbinom(applications, p),
        logit(p) <- a[gid] + d[did],
        a[gid] ~ dnorm(0, 1.5),
        d[did] ~ dnorm(0, 1.5)
    ),
    data = nwo_data,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.0 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.0 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.0 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.0 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.0 seconds.
    Total execution time: 0.6 seconds.
    
    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.1 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.1 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.1 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.1 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 0.5 seconds.
    



```R
par(bg = 'white')
plot(precis(total_effects_model, depth = 2))
plot(precis(indirect_effects_model, depth = 2))
```


    
![png](rethinking_ch11_output_226_0.png)
    



    
![png](rethinking_ch11_output_226_1.png)
    



```R
compare(indirect_effects_model, total_effects_model, func = WAIC)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>indirect_effects_model</th><td>128.2117</td><td>8.016974</td><td>0.000000</td><td>      NA</td><td>12.426389</td><td>0.6687232</td></tr>
	<tr><th scope=row>total_effects_model</th><td>129.6165</td><td>8.757184</td><td>1.404832</td><td>9.277178</td><td> 4.670875</td><td>0.3312768</td></tr>
</tbody>
</table>



From this, it looks like the two models do roughly as good a job explaining the discrimination. This is also borne out by the parameters above, where most of the $\delta$ parameters are clustered around zero, thus not contributing much to `logit(p)`. It seems like directly funding whole-university measures to get women to apply for the grants or provide them support or somesuch would be the most effective, rather than targeting individual departments.

### 4.19 11H5 Suppose that the NWO Grants sample has an unobserved confound that influences both the choice of discipline and the proavility of an award. One example of such a confound could be the career stage of each applicant. Suppose that in some disciplines, junior scholars apply for most of the grants. On other disciplines, scholars from all career stages compete. As a result, career stage influences discipline as well as the probability of being awarded a grant. Add these influences to your DAG from the previous proble. What happens now when you condition on discipline? Does it provide an un-confounded estimate of the direct path from gender to an award? Why or why not? Justify your answer with the backdoor criterion. If you have trouble thinks thins though, try simulating fake data, assuming your DAG is true. Then analyze it using the model from the previous problem. What do you conclude? Is it possible for gender to have a real direct causal influence but for a regression conditioning on both gender and discipline to suggest zero influence?


```R
par(bg = 'white')
dag <- dagitty('dag{
    G[pos="0,0"]
    D[pos="1,0"]
    A[pos="0.5,1"]
    S[unobserved, pos="1,1"]
    G -> D -> A; G -> A; A <- S -> D;
    }')
drawdag(dag)
```

    
![png](rethinking_ch11_output_230_0.png)
    


$D$ is now a collider, which means that by conditioning on it we open up a backdoor path through the career stage. If this DAG is correct, then there is no way (with the data that we have) to get an estimate of the indrect effects of gender on the admission rate. It is certainly possible for the gender to have a real effect, but to show no effect when conditioning on a collider.

### 4.20 11H6 The data in `data(Primates301)` are 301 primate species and associated measures. In this problem, you will consider how brain size is associated with social learning. There are three parts.
#### 4.20.1 a) Model the number of observations of `social_learning` for each species as a function of the log `brain` size. Use a Poisson distribution for the `social_learning` outcome variable. Interpret the resulting posterior.


```R
data(Primates301)
d <- Primates301
head(d)
```


<table class="dataframe">
<caption>A data.frame: 6 × 16</caption>
<thead>
	<tr><th></th><th scope=col>name</th><th scope=col>genus</th><th scope=col>species</th><th scope=col>subspecies</th><th scope=col>spp_id</th><th scope=col>genus_id</th><th scope=col>social_learning</th><th scope=col>research_effort</th><th scope=col>brain</th><th scope=col>body</th><th scope=col>group_size</th><th scope=col>gestation</th><th scope=col>weaning</th><th scope=col>longevity</th><th scope=col>sex_maturity</th><th scope=col>maternal_investment</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Allenopithecus_nigroviridis</td><td>Allenopithecus</td><td>nigroviridis</td><td>NA</td><td>1</td><td>1</td><td>0</td><td> 6</td><td>58.02</td><td>4655.00</td><td>40.0</td><td>    NA</td><td>106.15</td><td>276.0</td><td>     NA</td><td>    NA</td></tr>
	<tr><th scope=row>2</th><td>Allocebus_trichotis        </td><td>Allocebus     </td><td>trichotis   </td><td>NA</td><td>2</td><td>2</td><td>0</td><td> 6</td><td>   NA</td><td>  78.09</td><td> 1.0</td><td>    NA</td><td>    NA</td><td>   NA</td><td>     NA</td><td>    NA</td></tr>
	<tr><th scope=row>3</th><td>Alouatta_belzebul          </td><td>Alouatta      </td><td>belzebul    </td><td>NA</td><td>3</td><td>3</td><td>0</td><td>15</td><td>52.84</td><td>6395.00</td><td> 7.4</td><td>    NA</td><td>    NA</td><td>   NA</td><td>     NA</td><td>    NA</td></tr>
	<tr><th scope=row>4</th><td>Alouatta_caraya            </td><td>Alouatta      </td><td>caraya      </td><td>NA</td><td>4</td><td>3</td><td>0</td><td>45</td><td>52.63</td><td>5383.00</td><td> 8.9</td><td>185.92</td><td>323.16</td><td>243.6</td><td>1276.72</td><td>509.08</td></tr>
	<tr><th scope=row>5</th><td>Alouatta_guariba           </td><td>Alouatta      </td><td>guariba     </td><td>NA</td><td>5</td><td>3</td><td>0</td><td>37</td><td>51.70</td><td>5175.00</td><td> 7.4</td><td>    NA</td><td>    NA</td><td>   NA</td><td>     NA</td><td>    NA</td></tr>
	<tr><th scope=row>6</th><td>Alouatta_palliata          </td><td>Alouatta      </td><td>palliata    </td><td>NA</td><td>6</td><td>3</td><td>3</td><td>79</td><td>49.88</td><td>6250.00</td><td>13.1</td><td>185.42</td><td>495.60</td><td>300.0</td><td>1578.42</td><td>681.02</td></tr>
</tbody>
</table>




```R
summary(d)
```


                              name                genus            species   
     Allenopithecus_nigroviridis:  1   Macaca        : 24   fulvus     :  7  
     Allocebus_trichotis        :  1   Cercopithecus : 22   griseus    :  5  
     Alouatta_belzebul          :  1   Microcebus    : 17   troglodytes:  4  
     Alouatta_caraya            :  1   Lepilemur     : 16   cephus     :  3  
     Alouatta_guariba           :  1   Trachypithecus: 13   geoffroyi  :  3  
     Alouatta_palliata          :  1   Eulemur       : 12   nemestrina :  3  
     (Other)                    :295   (Other)       :197   (Other)    :276  
            subspecies      spp_id       genus_id     social_learning
     alaotrensis :  1   Min.   :  1   Min.   : 1.00   Min.   :  0.0  
     albifrons   :  1   1st Qu.: 76   1st Qu.:17.00   1st Qu.:  0.0  
     albocollaris:  1   Median :151   Median :36.00   Median :  0.0  
     atys        :  1   Mean   :151   Mean   :34.19   Mean   :  2.3  
     boliviensis :  1   3rd Qu.:226   3rd Qu.:48.00   3rd Qu.:  0.0  
     (Other)     : 29   Max.   :301   Max.   :68.00   Max.   :214.0  
     NA's        :267                                 NA's   :98     
     research_effort      brain             body             group_size    
     Min.   :  1.00   Min.   :  1.63   Min.   :    31.23   Min.   : 1.000  
     1st Qu.:  6.00   1st Qu.: 11.82   1st Qu.:   739.44   1st Qu.: 3.125  
     Median : 16.00   Median : 58.55   Median :  3553.50   Median : 7.500  
     Mean   : 38.76   Mean   : 68.49   Mean   :  6795.18   Mean   :13.263  
     3rd Qu.: 37.75   3rd Qu.: 86.20   3rd Qu.:  7465.00   3rd Qu.:18.225  
     Max.   :755.00   Max.   :491.27   Max.   :130000.00   Max.   :91.200  
     NA's   :115      NA's   :117      NA's   :63          NA's   :114     
       gestation         weaning         longevity       sex_maturity   
     Min.   : 59.99   Min.   :  40.0   Min.   : 103.0   Min.   : 283.2  
     1st Qu.:138.35   1st Qu.: 121.7   1st Qu.: 216.0   1st Qu.: 701.5  
     Median :166.03   Median : 234.2   Median : 301.2   Median :1427.2  
     Mean   :164.50   Mean   : 311.1   Mean   : 332.0   Mean   :1480.2  
     3rd Qu.:183.26   3rd Qu.: 388.8   3rd Qu.: 393.3   3rd Qu.:1894.1  
     Max.   :274.78   Max.   :1260.8   Max.   :1470.0   Max.   :5582.9  
     NA's   :161      NA's   :185      NA's   :181      NA's   :194     
     maternal_investment
     Min.   :  99.99    
     1st Qu.: 255.88    
     Median : 401.35    
     Mean   : 478.64    
     3rd Qu.: 592.22    
     Max.   :1492.30    
     NA's   :197        



```R
# first, let's filter out the rows with NA
# TODO actually filter out the NA values
filtered <- d[!is.na(d$brain) & !is.na(d$social_learning) & !is.na(d$research_effort), ]
# nrow(d) - nrow(filtered)
filtered
```


<table class="dataframe">
<caption>A data.frame: 150 × 16</caption>
<thead>
	<tr><th></th><th scope=col>name</th><th scope=col>genus</th><th scope=col>species</th><th scope=col>subspecies</th><th scope=col>spp_id</th><th scope=col>genus_id</th><th scope=col>social_learning</th><th scope=col>research_effort</th><th scope=col>brain</th><th scope=col>body</th><th scope=col>group_size</th><th scope=col>gestation</th><th scope=col>weaning</th><th scope=col>longevity</th><th scope=col>sex_maturity</th><th scope=col>maternal_investment</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Allenopithecus_nigroviridis</td><td>Allenopithecus</td><td>nigroviridis  </td><td>NA</td><td> 1</td><td> 1</td><td> 0</td><td>  6</td><td> 58.02</td><td>4655.00</td><td>40.00</td><td>    NA</td><td>106.15</td><td>276.0</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>3</th><td>Alouatta_belzebul          </td><td>Alouatta      </td><td>belzebul      </td><td>NA</td><td> 3</td><td> 3</td><td> 0</td><td> 15</td><td> 52.84</td><td>6395.00</td><td> 7.40</td><td>    NA</td><td>    NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>4</th><td>Alouatta_caraya            </td><td>Alouatta      </td><td>caraya        </td><td>NA</td><td> 4</td><td> 3</td><td> 0</td><td> 45</td><td> 52.63</td><td>5383.00</td><td> 8.90</td><td>185.92</td><td>323.16</td><td>243.6</td><td>1276.72</td><td> 509.08</td></tr>
	<tr><th scope=row>5</th><td>Alouatta_guariba           </td><td>Alouatta      </td><td>guariba       </td><td>NA</td><td> 5</td><td> 3</td><td> 0</td><td> 37</td><td> 51.70</td><td>5175.00</td><td> 7.40</td><td>    NA</td><td>    NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>6</th><td>Alouatta_palliata          </td><td>Alouatta      </td><td>palliata      </td><td>NA</td><td> 6</td><td> 3</td><td> 3</td><td> 79</td><td> 49.88</td><td>6250.00</td><td>13.10</td><td>185.42</td><td>495.60</td><td>300.0</td><td>1578.42</td><td> 681.02</td></tr>
	<tr><th scope=row>7</th><td>Alouatta_pigra             </td><td>Alouatta      </td><td>pigra         </td><td>NA</td><td> 7</td><td> 3</td><td> 0</td><td> 25</td><td> 51.13</td><td>8915.00</td><td> 5.50</td><td>185.92</td><td>    NA</td><td>240.0</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>8</th><td>Alouatta_sara              </td><td>Alouatta      </td><td>sara          </td><td>NA</td><td> 8</td><td> 3</td><td> 0</td><td>  4</td><td> 59.08</td><td>6611.04</td><td>   NA</td><td>    NA</td><td>    NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>9</th><td>Alouatta_seniculus         </td><td>Alouatta      </td><td>seniculus     </td><td>NA</td><td> 9</td><td> 3</td><td> 0</td><td> 82</td><td> 55.22</td><td>5950.00</td><td> 7.90</td><td>189.90</td><td>370.04</td><td>300.0</td><td>1690.22</td><td> 559.94</td></tr>
	<tr><th scope=row>10</th><td>Aotus_azarai               </td><td>Aotus         </td><td>azarai        </td><td>NA</td><td>10</td><td> 4</td><td> 0</td><td> 22</td><td> 20.67</td><td>1205.00</td><td> 4.10</td><td>    NA</td><td>229.69</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>14</th><td>Aotus_lemurinus            </td><td>Aotus         </td><td>lemurinus     </td><td>NA</td><td>14</td><td> 4</td><td> 0</td><td> 16</td><td> 16.30</td><td> 734.00</td><td>   NA</td><td>132.23</td><td> 74.57</td><td>216.0</td><td> 755.15</td><td> 206.80</td></tr>
	<tr><th scope=row>18</th><td>Aotus_trivirgatus          </td><td>Aotus         </td><td>trivirgatus   </td><td>NA</td><td>18</td><td> 4</td><td> 0</td><td> 58</td><td> 16.85</td><td> 989.00</td><td> 3.15</td><td>133.47</td><td> 76.21</td><td>303.6</td><td> 736.60</td><td> 209.68</td></tr>
	<tr><th scope=row>22</th><td>Arctocebus_calabarensis    </td><td>Arctocebus    </td><td>calabarensis  </td><td>NA</td><td>22</td><td> 6</td><td> 0</td><td>  1</td><td>  6.92</td><td> 309.00</td><td> 1.00</td><td>133.74</td><td>109.26</td><td>156.0</td><td> 298.91</td><td> 243.00</td></tr>
	<tr><th scope=row>23</th><td>Ateles_belzebuth           </td><td>Ateles        </td><td>belzebuth     </td><td>NA</td><td>23</td><td> 7</td><td> 0</td><td> 12</td><td>117.02</td><td>8167.00</td><td>14.50</td><td>138.20</td><td>    NA</td><td>336.0</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>24</th><td>Ateles_fusciceps           </td><td>Ateles        </td><td>fusciceps     </td><td>NA</td><td>24</td><td> 7</td><td> 0</td><td>  4</td><td>114.24</td><td>9025.00</td><td>   NA</td><td>224.70</td><td>482.70</td><td>288.0</td><td>1799.68</td><td> 707.40</td></tr>
	<tr><th scope=row>25</th><td>Ateles_geoffroyi           </td><td>Ateles        </td><td>geoffroyi     </td><td>NA</td><td>25</td><td> 7</td><td> 2</td><td> 58</td><td>105.09</td><td>7535.00</td><td>42.00</td><td>226.37</td><td>816.35</td><td>327.6</td><td>2104.57</td><td>1042.72</td></tr>
	<tr><th scope=row>26</th><td>Ateles_paniscus            </td><td>Ateles        </td><td>paniscus      </td><td>NA</td><td>26</td><td> 7</td><td> 0</td><td> 30</td><td>103.85</td><td>8280.00</td><td>20.00</td><td>228.18</td><td>805.41</td><td>453.6</td><td>2104.57</td><td>1033.59</td></tr>
	<tr><th scope=row>28</th><td>Avahi_laniger              </td><td>Avahi         </td><td>laniger       </td><td>NA</td><td>28</td><td> 8</td><td> 0</td><td> 10</td><td>  9.86</td><td>1207.00</td><td> 2.00</td><td>136.15</td><td>149.15</td><td>   NA</td><td>     NA</td><td> 285.30</td></tr>
	<tr><th scope=row>29</th><td>Avahi_occidentalis         </td><td>Avahi         </td><td>occidentalis  </td><td>NA</td><td>29</td><td> 8</td><td> 0</td><td>  6</td><td>  7.95</td><td> 801.00</td><td> 3.00</td><td>    NA</td><td>    NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>32</th><td>Bunopithecus_hoolock       </td><td>Bunopithecus  </td><td>hoolock       </td><td>NA</td><td>32</td><td>10</td><td> 0</td><td> 24</td><td>110.68</td><td>6728.00</td><td> 3.20</td><td>232.50</td><td>635.13</td><td>   NA</td><td>2689.08</td><td> 867.63</td></tr>
	<tr><th scope=row>33</th><td>Cacajao_calvus             </td><td>Cacajao       </td><td>calvus        </td><td>NA</td><td>33</td><td>11</td><td> 0</td><td> 11</td><td> 76.00</td><td>3165.00</td><td>23.70</td><td>180.00</td><td>339.29</td><td>324.0</td><td>1262.74</td><td> 519.29</td></tr>
	<tr><th scope=row>34</th><td>Cacajao_melanocephalus     </td><td>Cacajao       </td><td>melanocephalus</td><td>NA</td><td>34</td><td>11</td><td> 0</td><td>  8</td><td> 68.77</td><td>2935.00</td><td>30.00</td><td>    NA</td><td>    NA</td><td>216.0</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>40</th><td>Callimico_goeldii          </td><td>Callimico     </td><td>goeldii       </td><td>NA</td><td>40</td><td>13</td><td> 0</td><td> 43</td><td> 11.43</td><td> 484.00</td><td> 6.85</td><td>153.99</td><td> 66.53</td><td>214.8</td><td> 413.84</td><td> 220.52</td></tr>
	<tr><th scope=row>41</th><td>Callithrix_argentata       </td><td>Callithrix    </td><td>argentata     </td><td>NA</td><td>41</td><td>14</td><td> 0</td><td> 16</td><td>  7.95</td><td> 345.00</td><td> 9.50</td><td>    NA</td><td>    NA</td><td>201.6</td><td> 701.52</td><td>     NA</td></tr>
	<tr><th scope=row>46</th><td>Callithrix_jacchus         </td><td>Callithrix    </td><td>jacchus       </td><td>NA</td><td>46</td><td>14</td><td> 2</td><td>161</td><td>  7.24</td><td> 320.00</td><td> 8.55</td><td>144.00</td><td> 60.24</td><td>201.6</td><td> 455.99</td><td> 204.24</td></tr>
	<tr><th scope=row>50</th><td>Callithrix_pygmaea         </td><td>Callithrix    </td><td>pygmaea       </td><td>NA</td><td>50</td><td>14</td><td> 0</td><td> 36</td><td>  4.17</td><td> 116.00</td><td> 6.00</td><td>134.44</td><td> 90.73</td><td>181.2</td><td> 708.50</td><td> 225.17</td></tr>
	<tr><th scope=row>51</th><td>Cebus_albifrons            </td><td>Cebus         </td><td>albifrons     </td><td>NA</td><td>51</td><td>15</td><td> 1</td><td> 13</td><td> 65.45</td><td>2735.00</td><td>25.00</td><td>158.29</td><td>270.32</td><td>528.0</td><td>1501.69</td><td> 428.61</td></tr>
	<tr><th scope=row>52</th><td>Cebus_apella               </td><td>Cebus         </td><td>apella        </td><td>NA</td><td>52</td><td>15</td><td>17</td><td>249</td><td> 66.63</td><td>2936.00</td><td> 7.90</td><td>154.99</td><td>263.12</td><td>541.2</td><td>1760.81</td><td> 418.11</td></tr>
	<tr><th scope=row>53</th><td>Cebus_capucinus            </td><td>Cebus         </td><td>capucinus     </td><td>NA</td><td>53</td><td>15</td><td> 5</td><td> 60</td><td> 72.93</td><td>2861.00</td><td>18.15</td><td>161.06</td><td>514.07</td><td>657.6</td><td>2134.73</td><td> 675.13</td></tr>
	<tr><th scope=row>54</th><td>Cebus_olivaceus            </td><td>Cebus         </td><td>olivaceus     </td><td>NA</td><td>54</td><td>15</td><td> 0</td><td> 18</td><td> 69.84</td><td>2931.00</td><td>11.45</td><td>    NA</td><td>725.86</td><td>492.0</td><td>2525.48</td><td>     NA</td></tr>
	<tr><th scope=row>57</th><td>Cercocebus_galeritus       </td><td>Cercocebus    </td><td>galeritus     </td><td>NA</td><td>57</td><td>16</td><td> 0</td><td> 19</td><td> 99.07</td><td>7435.00</td><td>20.35</td><td>174.43</td><td>    NA</td><td>252.0</td><td>2735.94</td><td>     NA</td></tr>
	<tr><th scope=row>⋮</th><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td><td>⋮</td></tr>
	<tr><th scope=row>246</th><td>Pithecia_pithecia          </td><td>Pithecia      </td><td>pithecia   </td><td>NA       </td><td>246</td><td>53</td><td> 0</td><td> 28</td><td> 32.26</td><td> 1760</td><td> 2.70</td><td>161.13</td><td> 113.15</td><td>248.4</td><td>1089.37</td><td> 274.28</td></tr>
	<tr><th scope=row>248</th><td>Pongo_pygmaeus             </td><td>Pongo         </td><td>pygmaeus   </td><td>NA       </td><td>248</td><td>54</td><td>86</td><td>321</td><td>377.38</td><td>58542</td><td> 1.00</td><td>259.42</td><td>1088.80</td><td>720.0</td><td>3318.62</td><td>1348.22</td></tr>
	<tr><th scope=row>249</th><td>Presbytis_comata           </td><td>Presbytis     </td><td>comata     </td><td>NA       </td><td>249</td><td>55</td><td> 0</td><td> 11</td><td> 80.30</td><td> 6695</td><td> 7.05</td><td>    NA</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>250</th><td>Presbytis_melalophos       </td><td>Presbytis     </td><td>melalophos </td><td>NA       </td><td>250</td><td>55</td><td> 0</td><td>  6</td><td> 64.85</td><td> 6560</td><td>14.00</td><td>    NA</td><td>     NA</td><td>192.0</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>251</th><td>Procolobus_verus           </td><td>Procolobus    </td><td>verus      </td><td>NA       </td><td>251</td><td>56</td><td> 0</td><td>  3</td><td> 52.60</td><td> 4450</td><td> 6.30</td><td>167.84</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>254</th><td>Propithecus_diadema        </td><td>Propithecus   </td><td>diadema    </td><td>NA       </td><td>254</td><td>57</td><td> 0</td><td> 28</td><td> 39.80</td><td> 6130</td><td> 4.95</td><td>152.08</td><td> 256.27</td><td>   NA</td><td>1683.65</td><td> 408.35</td></tr>
	<tr><th scope=row>257</th><td>Propithecus_verreauxi      </td><td>Propithecus   </td><td>verreauxi  </td><td>NA       </td><td>257</td><td>57</td><td> 1</td><td> 41</td><td> 26.21</td><td> 2955</td><td> 6.30</td><td>149.77</td><td> 177.83</td><td>247.2</td><td> 943.94</td><td> 327.60</td></tr>
	<tr><th scope=row>259</th><td>Pygathrix_nemaeus          </td><td>Pygathrix     </td><td>nemaeus    </td><td>NA       </td><td>259</td><td>58</td><td> 0</td><td> 25</td><td> 91.41</td><td> 9720</td><td> 9.30</td><td>182.88</td><td>     NA</td><td>300.0</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>263</th><td>Rhinopithecus_roxellana    </td><td>Rhinopithecus </td><td>roxellana  </td><td>NA       </td><td>263</td><td>59</td><td> 0</td><td> 36</td><td>117.76</td><td>14750</td><td>65.00</td><td>199.34</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>266</th><td>Saguinus_fuscicollis       </td><td>Saguinus      </td><td>fuscicollis</td><td>NA       </td><td>266</td><td>61</td><td> 2</td><td> 81</td><td>  7.94</td><td>  401</td><td> 6.00</td><td>148.00</td><td>  90.10</td><td>294.0</td><td> 406.61</td><td> 238.10</td></tr>
	<tr><th scope=row>270</th><td>Saguinus_leucopus          </td><td>Saguinus      </td><td>leucopus   </td><td>NA       </td><td>270</td><td>61</td><td> 0</td><td>  3</td><td>  9.70</td><td>  525</td><td> 7.50</td><td>142.50</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>271</th><td>Saguinus_midas             </td><td>Saguinus      </td><td>midas      </td><td>NA       </td><td>271</td><td>61</td><td> 0</td><td> 17</td><td>  9.78</td><td>  563</td><td> 5.55</td><td>138.24</td><td>  69.60</td><td>184.8</td><td> 841.82</td><td> 207.84</td></tr>
	<tr><th scope=row>272</th><td>Saguinus_mystax            </td><td>Saguinus      </td><td>mystax     </td><td>NA       </td><td>272</td><td>61</td><td> 0</td><td> 46</td><td> 11.09</td><td>  584</td><td> 5.40</td><td>148.28</td><td>     NA</td><td>   NA</td><td> 556.85</td><td>     NA</td></tr>
	<tr><th scope=row>274</th><td>Saguinus_oedipus           </td><td>Saguinus      </td><td>oedipus    </td><td>NA       </td><td>274</td><td>61</td><td> 0</td><td>153</td><td>  9.76</td><td>  431</td><td> 7.05</td><td>166.49</td><td>  49.85</td><td>277.2</td><td> 680.38</td><td> 216.34</td></tr>
	<tr><th scope=row>277</th><td>Saimiri_oerstedii          </td><td>Saimiri       </td><td>oerstedii  </td><td>NA       </td><td>277</td><td>62</td><td> 1</td><td>  4</td><td> 25.07</td><td>  789</td><td>25.10</td><td>161.00</td><td> 362.93</td><td>   NA</td><td>     NA</td><td> 523.93</td></tr>
	<tr><th scope=row>278</th><td>Saimiri_sciureus           </td><td>Saimiri       </td><td>sciureus   </td><td>NA       </td><td>278</td><td>62</td><td> 1</td><td> 89</td><td> 24.14</td><td>  799</td><td>34.85</td><td>164.09</td><td> 177.41</td><td>324.0</td><td>1399.88</td><td> 341.50</td></tr>
	<tr><th scope=row>280</th><td>Semnopithecus_entellus     </td><td>Semnopithecus </td><td>entellus   </td><td>NA       </td><td>280</td><td>63</td><td> 2</td><td> 98</td><td>110.93</td><td>14742</td><td>19.00</td><td>197.70</td><td> 402.10</td><td>300.0</td><td>1497.64</td><td> 599.80</td></tr>
	<tr><th scope=row>281</th><td>Symphalangus_syndactylus   </td><td>Symphalangus  </td><td>syndactylus</td><td>NA       </td><td>281</td><td>64</td><td> 0</td><td> 40</td><td>123.50</td><td>11295</td><td> 3.80</td><td>230.66</td><td> 635.38</td><td>456.0</td><td>3788.23</td><td> 866.04</td></tr>
	<tr><th scope=row>282</th><td>Tarsius_bancanus           </td><td>Tarsius       </td><td>bancanus   </td><td>NA       </td><td>282</td><td>65</td><td> 0</td><td>  8</td><td>  3.16</td><td>  126</td><td> 1.00</td><td>125.84</td><td>  78.55</td><td>144.0</td><td> 658.68</td><td> 204.39</td></tr>
	<tr><th scope=row>283</th><td>Tarsius_dentatus           </td><td>Tarsius       </td><td>dentatus   </td><td>NA       </td><td>283</td><td>65</td><td> 0</td><td>  2</td><td>  3.00</td><td>  113</td><td> 1.00</td><td>    NA</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>285</th><td>Tarsius_syrichta           </td><td>Tarsius       </td><td>syrichta   </td><td>NA       </td><td>285</td><td>65</td><td> 0</td><td> 10</td><td>  3.36</td><td>  126</td><td> 1.00</td><td>177.99</td><td>  82.49</td><td>180.0</td><td>     NA</td><td> 260.48</td></tr>
	<tr><th scope=row>286</th><td>Theropithecus_gelada       </td><td>Theropithecus </td><td>gelada     </td><td>NA       </td><td>286</td><td>66</td><td> 0</td><td> 34</td><td>133.33</td><td>15350</td><td>10.00</td><td>178.64</td><td> 494.95</td><td>336.0</td><td>1894.11</td><td> 673.59</td></tr>
	<tr><th scope=row>288</th><td>Trachypithecus_cristatus   </td><td>Trachypithecus</td><td>cristatus  </td><td>NA       </td><td>288</td><td>67</td><td> 0</td><td>  8</td><td> 57.86</td><td> 6394</td><td>27.40</td><td>    NA</td><td> 362.93</td><td>373.2</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>291</th><td>Trachypithecus_geei        </td><td>Trachypithecus</td><td>geei       </td><td>NA       </td><td>291</td><td>67</td><td> 0</td><td>  7</td><td> 81.30</td><td>10150</td><td>11.00</td><td>    NA</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>293</th><td>Trachypithecus_johnii      </td><td>Trachypithecus</td><td>johnii     </td><td>NA       </td><td>293</td><td>67</td><td> 1</td><td>  9</td><td> 84.60</td><td>11600</td><td>10.00</td><td>    NA</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>295</th><td>Trachypithecus_obscurus    </td><td>Trachypithecus</td><td>obscurus   </td><td>NA       </td><td>295</td><td>67</td><td> 0</td><td>  6</td><td> 62.12</td><td> 7056</td><td>10.00</td><td>146.63</td><td> 362.93</td><td>300.0</td><td>     NA</td><td> 509.56</td></tr>
	<tr><th scope=row>296</th><td>Trachypithecus_phayrei     </td><td>Trachypithecus</td><td>phayrei    </td><td>NA       </td><td>296</td><td>67</td><td> 0</td><td> 16</td><td> 72.84</td><td> 7475</td><td>12.90</td><td>180.61</td><td> 305.87</td><td>   NA</td><td>     NA</td><td> 486.48</td></tr>
	<tr><th scope=row>297</th><td>Trachypithecus_pileatus    </td><td>Trachypithecus</td><td>pileatus   </td><td>NA       </td><td>297</td><td>67</td><td> 0</td><td>  5</td><td>103.64</td><td>11794</td><td> 8.50</td><td>    NA</td><td>     NA</td><td>   NA</td><td>     NA</td><td>     NA</td></tr>
	<tr><th scope=row>299</th><td>Trachypithecus_vetulus     </td><td>Trachypithecus</td><td>vetulus    </td><td>NA       </td><td>299</td><td>67</td><td> 0</td><td>  2</td><td> 61.29</td><td> 6237</td><td> 8.35</td><td>204.72</td><td> 245.78</td><td>276.0</td><td>1113.70</td><td> 450.50</td></tr>
	<tr><th scope=row>301</th><td>Varecia_variegata_variegata</td><td>Varecia       </td><td>variegata  </td><td>variegata</td><td>301</td><td>68</td><td> 0</td><td> 57</td><td> 32.12</td><td> 3575</td><td> 2.80</td><td>102.50</td><td>  90.73</td><td>384.0</td><td> 701.52</td><td> 193.23</td></tr>
</tbody>
</table>




```R
data <- list(
    s = filtered$social_learning,
    b = normalize(log(filtered$brain))
)

primate_social_learning_model_1 <- ulam(
    alist(
        s ~ dpois(lambda),
        log(lambda) <- a + b_b * b,
        a ~ dnorm(0, 1),
        b_b ~ dnorm(0, 0.5)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.1 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.1 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.1 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.1 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.1 seconds.
    Total execution time: 1.0 seconds.
    



```R
par(bg = 'white')
precis(primate_social_learning_model_1)
plot(precis(primate_social_learning_model_1))
```


<table class="dataframe">
<caption>A precis: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>-4.829520</td><td>0.2292752</td><td>-5.179700</td><td>-4.473379</td><td>1.020387</td><td>350.2590</td></tr>
	<tr><th scope=row>b_b</th><td> 8.722104</td><td>0.2864944</td><td> 8.266605</td><td> 9.166238</td><td>1.018082</td><td>344.9024</td></tr>
</tbody>
</table>




    
![png](rethinking_ch11_output_237_1.png)
    



```R
samples <- extract.samples(primate_social_learning_model_1)
lapply(samples, head)
```


<dl>
	<dt>$a</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 1 of type dbl</caption>
<tbody>
	<tr><td>-4.73463</td></tr>
	<tr><td>-4.98671</td></tr>
	<tr><td>-4.96773</td></tr>
	<tr><td>-5.02689</td></tr>
	<tr><td>-4.64126</td></tr>
	<tr><td>-4.74712</td></tr>
</tbody>
</table>
</dd>
	<dt>$b_b</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 1 of type dbl</caption>
<tbody>
	<tr><td>8.68895</td></tr>
	<tr><td>8.83919</td></tr>
	<tr><td>8.94851</td></tr>
	<tr><td>8.94834</td></tr>
	<tr><td>8.47234</td></tr>
	<tr><td>8.64739</td></tr>
</tbody>
</table>
</dd>
	<dt>$lambda</dt>
		<dd><table class="dataframe">
<caption>A matrix: 6 × 150 of type dbl</caption>
<tbody>
	<tr><td>2.02264</td><td>1.75419</td><td>1.74359</td><td>1.69690</td><td>1.60678</td><td>1.66849</td><td>2.07917</td><td>1.87590</td><td>0.420178</td><td>0.292666</td><td>⋯</td><td>0.0264303</td><td>7.17974</td><td>2.01415</td><td>3.38063</td><td>3.59177</td><td>2.24424</td><td>2.85982</td><td>4.89256</td><td>2.19874</td><td>0.822076</td></tr>
	<tr><td>1.72698</td><td>1.49409</td><td>1.48490</td><td>1.44446</td><td>1.36646</td><td>1.41987</td><td>1.77609</td><td>1.59961</td><td>0.349141</td><td>0.241671</td><td>⋯</td><td>0.0209362</td><td>6.26600</td><td>1.71961</td><td>2.91222</td><td>3.09734</td><td>1.91963</td><td>2.45645</td><td>4.24168</td><td>1.88005</td><td>0.691066</td></tr>
	<tr><td>1.88473</td><td>1.62765</td><td>1.61752</td><td>1.57293</td><td>1.48697</td><td>1.54582</td><td>1.93900</td><td>1.74407</td><td>0.373574</td><td>0.257409</td><td>⋯</td><td>0.0216351</td><td>6.94825</td><td>1.87659</td><td>3.19885</td><td>3.40478</td><td>2.09773</td><td>2.69255</td><td>4.68087</td><td>2.05394</td><td>0.745697</td></tr>
	<tr><td>1.77627</td><td>1.53399</td><td>1.52444</td><td>1.48241</td><td>1.40141</td><td>1.45687</td><td>1.82742</td><td>1.64371</td><td>0.352087</td><td>0.242606</td><td>⋯</td><td>0.0203918</td><td>6.54823</td><td>1.76859</td><td>3.01473</td><td>3.20881</td><td>1.97700</td><td>2.53758</td><td>4.41142</td><td>1.93574</td><td>0.702797</td></tr>
	<tr><td>1.93902</td><td>1.68765</td><td>1.67770</td><td>1.63388</td><td>1.54922</td><td>1.60721</td><td>1.99184</td><td>1.80173</td><td>0.418900</td><td>0.294419</td><td>⋯</td><td>0.0282311</td><td>6.66893</td><td>1.93108</td><td>3.19963</td><td>3.39433</td><td>2.14589</td><td>2.71802</td><td>4.58813</td><td>2.10346</td><td>0.805977</td></tr>
	<tr><td>1.94624</td><td>1.68908</td><td>1.67892</td><td>1.63417</td><td>1.54779</td><td>1.60695</td><td>2.00037</td><td>1.80570</td><td>0.407357</td><td>0.284228</td><td>⋯</td><td>0.0259651</td><td>6.86680</td><td>1.93811</td><td>3.24495</td><td>3.44662</td><td>2.15839</td><td>2.74724</td><td>4.68790</td><td>2.11484</td><td>0.794437</td></tr>
</tbody>
</table>
</dd>
</dl>




```R
posterior_lambda_means <- apply(samples$lambda, 2, mean)
posterior_lambda_ranges <- apply(samples$lambda, 2, PI)
head(posterior_lambda_ranges)
```


<table class="dataframe">
<caption>A matrix: 2 × 150 of type dbl</caption>
<tbody>
	<tr><th scope=row>5%</th><td>1.690990</td><td>1.460057</td><td>1.450851</td><td>1.410556</td><td>1.332998</td><td>1.386437</td><td>1.739943</td><td>1.564812</td><td>0.3272314</td><td>0.2235456</td><td>⋯</td><td>0.01786677</td><td>6.218723</td><td>1.683424</td><td>2.881245</td><td>3.071093</td><td>1.882456</td><td>2.424270</td><td>4.215132</td><td>1.843428</td><td>0.6628805</td></tr>
	<tr><th scope=row>94%</th><td>2.080967</td><td>1.814693</td><td>1.804350</td><td>1.758475</td><td>1.668270</td><td>1.729674</td><td>2.137675</td><td>1.933338</td><td>0.4605060</td><td>0.3258945</td><td>⋯</td><td>0.03270554</td><td>7.214618</td><td>2.072491</td><td>3.427656</td><td>3.634447</td><td>2.302012</td><td>2.912698</td><td>4.929703</td><td>2.256809</td><td>0.8759700</td></tr>
</tbody>
</table>




```R
plot_df <- data.frame(
    name = filtered$name,
    s = data$s,
    b = data$b,
    mean_lambda = posterior_lambda_means,
    lambda_lower = posterior_lambda_ranges[1, ],
    lambda_upper = posterior_lambda_ranges[2, ]
)
```


```R
ggplot(plot_df, aes(name, s)) +
    geom_point() +
    geom_point(data = plot_df, mapping = aes(name, mean_lambda), colour = 'blue', position = position_dodge(width = 0.1)) +
    geom_linerange(mapping = aes(name, ymin = lambda_lower, ymax = lambda_upper), colour = 'blue', position = position_dodge(width = 0.1)) +
    theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))
```


    
![png](rethinking_ch11_output_241_0.png)
    


Generally this does a good job, but there are definitely a few outliers where our prediction is badly wrong.

#### 4.20.2 b) Some species are studied much more than others. So the number of reported instances of `social_learning` could be a product od reearch effort. Use the `research_effort` variable, specifically its logarithm, as an additional predictor variable. Interpret the coefficient for lof `research_effort`. How does this model differ from the previous one?


```R
data <- list(
    s = filtered$social_learning,
    b = normalize(log(filtered$brain)),
    e = log(filtered$research_effort)
)

primate_social_learning_model_2 <- ulam(
    alist(
        s ~ dpois(lambda),
        log(lambda) <- a + b_b * b + b_e * e,
        a ~ dnorm(0, 1),
        c(b_b, b_e) ~ dnorm(0, 0.5)
    ),
    data = data,
    chains = 4,
    log_lik = TRUE
)
```

    Running MCMC with 4 sequential chains, with 1 thread(s) per chain...
    
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
    Chain 1 finished in 0.3 seconds.
    Chain 2 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 2 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 2 finished in 0.3 seconds.
    Chain 3 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 3 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 3 finished in 0.3 seconds.
    Chain 4 Iteration:   1 / 1000 [  0%]  (Warmup) 
    Chain 4 Iteration: 100 / 1000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 200 / 1000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 300 / 1000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 400 / 1000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 500 / 1000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 501 / 1000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 600 / 1000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 700 / 1000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 800 / 1000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 900 / 1000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 1000 / 1000 [100%]  (Sampling) 
    Chain 4 finished in 0.3 seconds.
    
    All 4 chains finished successfully.
    Mean chain execution time: 0.3 seconds.
    Total execution time: 1.6 seconds.
    



```R
samples <- extract.samples(primate_social_learning_model_2)
posterior_lambda_means <- apply(samples$lambda, 2, mean)
posterior_lambda_ranges <- apply(samples$lambda, 2, PI)

plot_df <- data.frame(
    name = filtered$name,
    s = data$s,
    mean_lambda = posterior_lambda_means,
    lambda_lower = posterior_lambda_ranges[1, ],
    lambda_upper = posterior_lambda_ranges[2, ]
)
ggplot(plot_df, aes(name, s)) +
    geom_point() +
    geom_point(data = plot_df, mapping = aes(name, mean_lambda), colour = 'blue', position = position_dodge(width = 0.1)) +
    geom_linerange(mapping = aes(name, ymin = lambda_lower, ymax = lambda_upper), colour = 'blue', position = position_dodge(width = 0.1)) +
    theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))
```


    
![png](rethinking_ch11_output_244_0.png)
    


This generally looks like it's doing a better job.


```R
compare(primate_social_learning_model_1, primate_social_learning_model_2, func = WAIC)
```


<table class="dataframe">
<caption>A compareIC: 2 × 6</caption>
<thead>
	<tr><th></th><th scope=col>WAIC</th><th scope=col>SE</th><th scope=col>dWAIC</th><th scope=col>dSE</th><th scope=col>pWAIC</th><th scope=col>weight</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>primate_social_learning_model_2</th><td> 557.4717</td><td>154.3577</td><td>   0.000</td><td>      NA</td><td> 45.69722</td><td> 1.000000e+00</td></tr>
	<tr><th scope=row>primate_social_learning_model_1</th><td>1664.1810</td><td>701.8233</td><td>1106.709</td><td>658.5704</td><td>173.79219</td><td>4.798904e-241</td></tr>
</tbody>
</table>



This is borne out by the comparison above, where the second model is quite a bit better than the first one. 

In terms of interpretation of the coefficient of the research effort, let's first find what it is!


```R
par(bg = 'white')
precis(primate_social_learning_model_2)
plot(precis(primate_social_learning_model_2))
```


<table class="dataframe">
<caption>A precis: 3 × 6</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>5.5%</th><th scope=col>94.5%</th><th scope=col>rhat</th><th scope=col>ess_bulk</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>a</th><td>-7.010041</td><td>0.27039785</td><td>-7.4431458</td><td>-6.604825</td><td>1.001830</td><td>522.1280</td></tr>
	<tr><th scope=row>b_e</th><td> 1.660250</td><td>0.06154162</td><td> 1.5655051</td><td> 1.761439</td><td>1.003220</td><td>509.5735</td></tr>
	<tr><th scope=row>b_b</th><td> 1.122993</td><td>0.27859917</td><td> 0.6694367</td><td> 1.572004</td><td>1.002796</td><td>679.1137</td></tr>
</tbody>
</table>




    
![png](rethinking_ch11_output_248_1.png)
    


Just like any regression coefficient, this means that a 1-unit increase in log of research effort is associated with a 1.65 ish unit increase in log lambda. Alternatively, a 1-unit increase in log research effort is associated with an $e^{1.65} \approx 5.2$ increase in observed social learning.


```R
exp(1.65)
```


5.20697982717985


#### 4.20.3 c) Draw a DAG to represent how you think the variables `social_learning`, `brain`, and `research_effort` interact. Justify the DAG with the measures associations in the two models above (and any other models you used).


```R
par(bg = "white")
dag <- dagitty('dag{
    S[pos="0,0"]
    B[pos="-0.5,0.5"]
    R[pos="0.5,0.5"]
    B -> S <- R; B -> R;
}')
drawdag(dag)
```


    
![png](rethinking_ch11_output_252_0.png)
    



```R
impliedConditionalIndependencies(dag)
```

This implies that there should be no independencies, which is in fact what we see if we look at the parameters in the second model (conditioning on both the brain size and researcher effort):


```R
par(bg = 'white')
plot(precis(primate_social_learning_model_2))
```


    
![png](rethinking_ch11_output_255_0.png)
    


This means that our measured associations are consistent with the DAG above.