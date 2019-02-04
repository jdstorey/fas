\providecommand{\E}{\operatorname{E}}
\providecommand{\V}{\operatorname{Var}}
\providecommand{\Cov}{\operatorname{Cov}}
\providecommand{\se}{\operatorname{se}}
\providecommand{\logit}{\operatorname{logit}}
\providecommand{\iid}{\; \stackrel{\text{iid}}{\sim}\;}
\providecommand{\asim}{\; \stackrel{.}{\sim}\;}
\providecommand{\xs}{x_1, x_2, \ldots, x_n}
\providecommand{\Xs}{X_1, X_2, \ldots, X_n}
\providecommand{\bB}{\boldsymbol{B}}
\providecommand{\bb}{\boldsymbol{\beta}}
\providecommand{\bx}{\boldsymbol{x}}
\providecommand{\bX}{\boldsymbol{X}}
\providecommand{\by}{\boldsymbol{y}}
\providecommand{\bY}{\boldsymbol{Y}}
\providecommand{\bz}{\boldsymbol{z}}
\providecommand{\bZ}{\boldsymbol{Z}}
\providecommand{\be}{\boldsymbol{e}}
\providecommand{\bE}{\boldsymbol{E}}
\providecommand{\bs}{\boldsymbol{s}}
\providecommand{\bS}{\boldsymbol{S}}
\providecommand{\bP}{\boldsymbol{P}}
\providecommand{\bI}{\boldsymbol{I}}
\providecommand{\bD}{\boldsymbol{D}}
\providecommand{\bd}{\boldsymbol{d}}
\providecommand{\bW}{\boldsymbol{W}}
\providecommand{\bw}{\boldsymbol{w}}
\providecommand{\bM}{\boldsymbol{M}}
\providecommand{\bPhi}{\boldsymbol{\Phi}}
\providecommand{\bphi}{\boldsymbol{\phi}}
\providecommand{\bN}{\boldsymbol{N}}
\providecommand{\bR}{\boldsymbol{R}}
\providecommand{\bu}{\boldsymbol{u}}
\providecommand{\bU}{\boldsymbol{U}}
\providecommand{\bv}{\boldsymbol{v}}
\providecommand{\bV}{\boldsymbol{V}}
\providecommand{\bO}{\boldsymbol{0}}
\providecommand{\bOmega}{\boldsymbol{\Omega}}
\providecommand{\bLambda}{\boldsymbol{\Lambda}}
\providecommand{\bSig}{\boldsymbol{\Sigma}}
\providecommand{\bSigma}{\boldsymbol{\Sigma}}
\providecommand{\bt}{\boldsymbol{\theta}}
\providecommand{\bT}{\boldsymbol{\Theta}}
\providecommand{\bpi}{\boldsymbol{\pi}}
\providecommand{\argmax}{\text{argmax}}
\providecommand{\KL}{\text{KL}}
\providecommand{\fdr}{{\rm FDR}}
\providecommand{\pfdr}{{\rm pFDR}}
\providecommand{\mfdr}{{\rm mFDR}}
\providecommand{\bh}{\hat}
\providecommand{\dd}{\lambda}
\providecommand{\q}{\operatorname{q}}





# (PART) Statistical Models {-}

# Types of Models

## Probabilistic Models

So far we have covered inference of paramters that quantify a population of interest. 

This is called inference of probabilistic models.

## Multivariate Models

Some of the probabilistic models we considered involve calculating conditional probabilities such as $\Pr(\bZ | \bX; \bt)$ or $\Pr(\bt | \bX)$.  

It is often the case that we would like to build a model that *explains the variation of one variable in terms of other variables*.  **Statistical modeling** typically refers to this goal.

## Variables

Let's suppose our does comes in the form $(\bX_1, Y_1), (\bX_2, Y_2), \ldots, (\bX_n, Y_n) \sim F$.

We will call $\bX_i = (X_{i1}, X_{i2}, \ldots, X_{ip}) \in \mathbb{R}_{1 \times p}$ the **explanatory variables** and $Y_i \in \mathbb{R}$ the **dependent variable** or **response variable**.

We can collect all variables as matrices

$$ \bY_{n \times 1} \ \mbox{ and } \ \bX_{n \times p}$$

where each row is a unique observation.

## Statistical Model

Statistical models are concerned with *how* variables are dependent.  The most general model would be to infer

$$
\Pr(Y | \bX) = h(\bX)
$$

where we would specifically study the form of $h(\cdot)$ to understand how $Y$ is dependent on $\bX$. 

A more modest goal is to infer the transformed conditional expecation

$$
g\left(\E[Y | \bX]\right) = h(\bX)
$$

which sometimes leads us back to an estimate of $\Pr(Y | \bX)$.

## Parametric vs Nonparametric

A **parametric** model is a pre-specified form of $h(X)$ whose terms can be characterized by a formula and interpreted. This usually involves parameters on which inference can be performed, such as coefficients in a linear model.

A **nonparametric** model is a data-driven form of $h(X)$ that is often very flexible and is not easily expressed or intepreted.  A nonparametric model often does not include parameters on which we can do inference.

## Simple Linear Regression

For random variables $(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)$, **simple linear regression** estimates the model

$$
Y_i  = \beta_1 + \beta_2 X_i + E_i
$$

where $\E[E_i] = 0$, $\V(E_i) = \sigma^2$, and $\Cov(E_i, E_j) = 0$ for all $1 \leq i, j \leq n$ and $i \not= j$.

Note that in this model $\E[Y | X] = \beta_1 + \beta_2 X.$

## Ordinary Least Squares

**Ordinary least squares** (OLS) estimates the model

$$
\begin{aligned}
Y_i & = \beta_1 X_{i1} + \beta_2 X_{i2} + \ldots + \beta_p X_{ip} + E_i \\
 & = \bX_i \bb + E_i
\end{aligned}
$$

where ${\rm E}[E_i] = 0$, ${\rm Var}(E_i) = \sigma^2$, and $\Cov(E_i, E_j) = 0$ for all $1 \leq i, j \leq n$ and $i \not= j$.

Note that typically $X_{i1} = 1$ for all $i$ so that $\beta_1 X_{i1} = \beta_1$ serves as the intercept.

## Generalized Least Squares

**Generalized least squares** (GLS) assumes the same model as OLS, except it allows for **heteroskedasticity** and **covariance** among the $E_i$.  Specifically, it is assumed that $\bE = (E_1, \ldots, E_n)^T$ is distributed as

$$
\bE_{n \times 1} \sim (\boldsymbol{0}, \bSig)
$$
where $\boldsymbol{0}$ is the expected value $\bSig = (\sigma_{ij})$ is the $n \times n$ symmetric covariance matrix.

## Matrix Form of Linear Models

We can write the models as

$$
\bY_{n \times 1} = \bX_{n \times p} \bb_{p \times 1} + \bE_{n \times 1}
$$

where simple linear regression, OLS, and GLS differ in the value of $p$ or the distribution of the $E_i$.  We can also write the conditional expecation and covariance as

$$
\E[\bY | \bX] = \bX \bb, \ \Cov(\bY | \bX) = \bSig.
$$

## Least Squares Regression

In simple linear regression, OLS, and GLS, the $\bb$ parameters are fit by minimizing the sum of squares between $\bY$ and $\bX \bb$.  

Fitting these models by "least squares" satisfies two types of optimality:

1.  [Gauss-Markov Theorem](https://en.wikipedia.org/wiki/Gauss–Markov_theorem)
2.  [Maximum likelihood estimate](https://en.wikipedia.org/wiki/Ordinary_least_squares#Maximum_likelihood) when in addition $\bE \sim \mbox{MVN}_n(\boldsymbol{0}, \bSig)$

Details will follow on these.

## Generalized Linear Models

The generalized linear model (GLM) builds from OLS and GLS to allow the response variable to be distributed according to an exponential family distribution.  Suppose that $\eta(\theta)$ is function of the expected value into the natural parameter.  The estimated model is

$$
\eta\left(\E[Y | \bX]\right) = \bX \bb
$$

which is fit by maximized likelihood estimation.



## Generalized Additive Models

Next week, we will finally arrive at inferring semiparametric models where $Y | \bX$ is distributed according to an exponential family distribution.  The models, which are called **generalized additive models** (GAMs), will be of the form

$$
\eta\left(\E[Y | \bX]\right) = \sum_{j=1}^p \sum_{k=1}^d h_k(X_{j})
$$

where $\eta$ is the canonical link function and the $h_k(\cdot)$ functions are very flexible.

## Some Trade-offs

There are several important trade-offs encountered in statistical modeling:

- Bias vs variance
- Accuracy vs computational time
- Flexibility vs intepretability

These are not mutually exclusive phenomena.

## Bias and Variance

Suppose we estimate $Y = h(\bX) + E$ by some $\hat{Y} = \hat{h}(\bX)$.  The following bias-variance trade-off exists:

$$
\begin{aligned}
\E\left[\left(Y - \hat{Y}\right)^2\right] & = {\rm E}\left[\left(h(\bX) + E - \hat{h}(\bX)\right)^2\right] \\
\ & = {\rm E}\left[\left(h(\bX) - \hat{h}(\bX)\right)^2\right] + {\rm Var}(E) \\
\ & = \left(h(\bX) - {\rm E}[\hat{h}(\bX)]\right)^2 + {\rm Var}\left(\hat{h}(\bX)\right)^2 + {\rm Var}(E) \\ 
\ & = \mbox{bias}^2 + \mbox{variance} + {\rm Var}(E)
\end{aligned}
$$

# Motivating Examples

## Sample Correlation

Least squares regression "modelizes" correlation.  Suppose we observe $n$ pairs of data $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$. Their sample correlation is

\begin{eqnarray}
r_{xy} & = & \frac{\sum_{i=1}^n (x_i - \overline{x}) (y_i - \overline{y})}{\sqrt{\sum_{i=1}^n (x_i - \overline{x})^2 \sum_{i=1}^n (y_i - \overline{y})^2}} \\
\ & = & \frac{\sum_{i=1}^n (x_i - \overline{x}) (y_i - \overline{y})}{(n-1) s_x s_y}
\end{eqnarray}

where $s_x$ and $s_y$ are the sample standard deviations of each measured variable.

## Example: Hand Size Vs. Height


```r
> library("MASS")
> data("survey", package="MASS")
> head(survey)
     Sex Wr.Hnd NW.Hnd W.Hnd    Fold Pulse    Clap Exer Smoke Height
1 Female   18.5   18.0 Right  R on L    92    Left Some Never 173.00
2   Male   19.5   20.5  Left  R on L   104    Left None Regul 177.80
3   Male   18.0   13.3 Right  L on R    87 Neither None Occas     NA
4   Male   18.8   18.9 Right  R on L    NA Neither None Never 160.00
5   Male   20.0   20.0 Right Neither    35   Right Some Never 165.00
6 Female   18.0   17.7 Right  L on R    64   Right Some Never 172.72
       M.I    Age
1   Metric 18.250
2 Imperial 17.583
3     <NA> 16.917
4   Metric 20.333
5   Metric 23.667
6 Imperial 21.000
```



```r
> ggplot(data = survey, mapping=aes(x=Wr.Hnd, y=Height)) +
+   geom_point() + geom_vline(xintercept=mean(survey$Wr.Hnd, na.rm=TRUE)) +
+   geom_hline(yintercept=mean(survey$Height, na.rm=TRUE))
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-3-1.png" width="672" style="display: block; margin: auto;" />

## Cor. of Hand Size and Height


```r
> cor.test(x=survey$Wr.Hnd, y=survey$Height)

	Pearson's product-moment correlation

data:  survey$Wr.Hnd and survey$Height
t = 10.792, df = 206, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.5063486 0.6813271
sample estimates:
      cor 
0.6009909 
```

## L/R Hand Sizes


```r
> ggplot(data = survey) +
+   geom_point(aes(x=Wr.Hnd, y=NW.Hnd))
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-5-1.png" width="672" style="display: block; margin: auto;" />

## Correlation of Hand Sizes


```r
> cor.test(x=survey$Wr.Hnd, y=survey$NW.Hnd)

	Pearson's product-moment correlation

data:  survey$Wr.Hnd and survey$NW.Hnd
t = 45.712, df = 234, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.9336780 0.9597816
sample estimates:
      cor 
0.9483103 
```

## Davis Data


```r
> library("car")
> data("Davis", package="car")
Warning in data("Davis", package = "car"): data set 'Davis' not found
```


```r
> htwt <- tbl_df(Davis)
> htwt[12,c(2,3)] <- htwt[12,c(3,2)]
> head(htwt)
# A tibble: 6 x 5
  sex   weight height repwt repht
  <fct>  <int>  <int> <int> <int>
1 M         77    182    77   180
2 F         58    161    51   159
3 F         53    161    54   158
4 M         68    177    70   175
5 F         59    157    59   155
6 M         76    170    76   165
```

## Height and Weight


```r
> ggplot(htwt) + 
+   geom_point(aes(x=height, y=weight, color=sex), size=2, alpha=0.5) +
+   scale_color_manual(values=c("red", "blue"))
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-9-1.png" width="672" style="display: block; margin: auto;" />

## Correlation of Height and Weight


```r
> cor.test(x=htwt$height, y=htwt$weight)

	Pearson's product-moment correlation

data:  htwt$height and htwt$weight
t = 17.04, df = 198, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.7080838 0.8218898
sample estimates:
      cor 
0.7710743 
```

## Correlation Among Females


```r
> htwt %>% filter(sex=="F") %>%  
+   cor.test(~ height + weight, data = .)

	Pearson's product-moment correlation

data:  height and weight
t = 6.2801, df = 110, p-value = 6.922e-09
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.3627531 0.6384268
sample estimates:
      cor 
0.5137293 
```

## Correlation Among Males


```r
> htwt %>% filter(sex=="M") %>%  
+   cor.test(~ height + weight, data = .)

	Pearson's product-moment correlation

data:  height and weight
t = 5.9388, df = 86, p-value = 5.922e-08
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.3718488 0.6727460
sample estimates:
      cor 
0.5392906 
```

Why are the stratified correlations lower?



# Simple Linear Regression

## Definition

For random variables $(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)$, **simple linear regression** estimates the model

$$
Y_i  = \beta_1 + \beta_2 X_i + E_i
$$

where $\E[E_i] = 0$, $\V(E_i) = \sigma^2$, and $\Cov(E_i, E_j) = 0$ for all $1 \leq i, j \leq n$ and $i \not= j$.

## Rationale

- **Least squares linear regression** is one of the simplest and most useful modeling systems for building a model that explains the variation of one variable in terms of other variables.

- It is simple to fit, it satisfies some optimality criteria, and it is straightforward to check assumptions on the data so that statistical inference can be performed.

## Setup

- Suppose that we have observed $n$ pairs of data $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$.

- **Least squares linear regression** models variation of the **response variable** $y$ in terms of the **explanatory variable** $x$ in the form of $\beta_1 + \beta_2 x$, where $\beta_1$ and $\beta_2$ are chosen to satisfy a least squares optimization.

## Line Minimizing Squared Error

The least squares regression line is formed from the value of $\beta_1$ and $\beta_2$ that minimize:

$$\sum_{i=1}^n \left( y_i - \beta_1 - \beta_2 x_i \right)^2.$$

For a given set of data, there is a unique solution to this minimization as long as there are at least two unique values among $x_1, x_2, \ldots, x_n$.  

Let $\hat{\beta_1}$ and $\hat{\beta_2}$ be the values that minimize this sum of squares.

## Least Squares Solution

These values are:

$$\hat{\beta}_2 = r_{xy} \frac{s_y}{s_x}$$

$$\hat{\beta}_1 = \overline{y} - \hat{\beta}_2 \overline{x}$$

These values have a useful interpretation.

## Visualizing Least Squares Line

<img src="stat_models_1_files/figure-html/unnamed-chunk-13-1.png" width="672" style="display: block; margin: auto;" />

## Example: Height and Weight


```r
> ggplot(data=htwt, mapping=aes(x=height, y=weight)) + 
+   geom_point(size=2, alpha=0.5) +
+   geom_smooth(method="lm", se=FALSE, formula=y~x)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-14-1.png" width="672" style="display: block; margin: auto;" />

## Calculate the Line Directly


```r
> beta2 <- cor(htwt$height, htwt$weight) * 
+                sd(htwt$weight) / sd(htwt$height)
> beta2
[1] 1.150092
> 
> beta1 <- mean(htwt$weight) - beta2 * mean(htwt$height)
> beta1
[1] -130.9104
> 
> yhat <- beta1 + beta2 * htwt$height
```

## Plot the Line


```r
> df <- data.frame(htwt, yhat=yhat)
> ggplot(data=df) + geom_point(aes(x=height, y=weight), size=2, alpha=0.5) +
+   geom_line(aes(x=height, y=yhat), color="blue", size=1.2)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-16-1.png" width="672" style="display: block; margin: auto;" />

## Observed Data, Fits, and Residuals

We observe data $(x_1, y_1), \ldots, (x_n, y_n)$. Note that we only observe $X_i$ and $Y_i$ from the generative model $Y_i = \beta_1 + \beta_2 X_i + E_i$.

We calculate fitted values and observed residuals:

$$\hat{y}_i = \hat{\beta}_1 + \hat{\beta}_2 x_i$$

$$\hat{e}_i = y_i - \hat{y}_i$$

By construction, it is the case that $\sum_{i=1}^n \hat{e}_i = 0$.

## Proportion of Variation Explained

The proportion of variance explained by the fitted model is called $R^2$ or $r^2$.  It is calculated by:

$$r^2 = \frac{s^2_{\hat{y}}}{s^2_{y}}$$

# `lm()` Function in R

## Calculate the Line in R

The syntax for a model in R is 

```response variable ~ explanatory variables``` 

where the `explanatory variables` component can involve several types of terms.


```r
> myfit <- lm(weight ~ height, data=htwt)
> myfit

Call:
lm(formula = weight ~ height, data = htwt)

Coefficients:
(Intercept)       height  
    -130.91         1.15  
```

## An `lm` Object is a List


```r
> class(myfit)
[1] "lm"
> is.list(myfit)
[1] TRUE
> names(myfit)
 [1] "coefficients"  "residuals"     "effects"       "rank"         
 [5] "fitted.values" "assign"        "qr"            "df.residual"  
 [9] "xlevels"       "call"          "terms"         "model"        
```

## From the R Help

> `lm` returns an object of class "lm" or for multiple responses of class c("mlm", "lm").

> The functions `summary` and `anova` are used to obtain and print a summary and analysis of variance table of the results. The generic accessor functions coefficients, effects, fitted.values and residuals extract various useful features of the value returned by `lm`.

## Some of the List Items

These are some useful items to access from the `lm` object:

- `coefficients`: a named vector of coefficients
- `residuals`:	the residuals, that is response minus fitted values.
- `fitted.values`: the fitted mean values.
- `df.residual`: the residual degrees of freedom.
- `call`: the matched call.
- `model`: if requested (the default), the model frame used.

## `summary()`


```r
> summary(myfit)

Call:
lm(formula = weight ~ height, data = htwt)

Residuals:
    Min      1Q  Median      3Q     Max 
-19.658  -5.381  -0.555   4.807  42.894 

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept) -130.91040   11.52792  -11.36   <2e-16 ***
height         1.15009    0.06749   17.04   <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 8.505 on 198 degrees of freedom
Multiple R-squared:  0.5946,	Adjusted R-squared:  0.5925 
F-statistic: 290.4 on 1 and 198 DF,  p-value: < 2.2e-16
```

## `summary()` List Elements


```r
> mysummary <- summary(myfit)
> names(mysummary)
 [1] "call"          "terms"         "residuals"     "coefficients" 
 [5] "aliased"       "sigma"         "df"            "r.squared"    
 [9] "adj.r.squared" "fstatistic"    "cov.unscaled" 
```

## Using `tidy()`


```r
> library(broom)
> tidy(myfit)
# A tibble: 2 x 5
  term        estimate std.error statistic  p.value
  <chr>          <dbl>     <dbl>     <dbl>    <dbl>
1 (Intercept)  -131.     11.5        -11.4 2.44e-23
2 height          1.15    0.0675      17.0 1.12e-40
```

## Proportion of Variation Explained

The proportion of variance explained by the fitted model is called $R^2$ or $r^2$.  It is calculated by:

$$r^2 = \frac{s^2_{\hat{y}}}{s^2_{y}}$$


```r
> summary(myfit)$r.squared
[1] 0.5945555
> 
> var(myfit$fitted.values)/var(htwt$weight)
[1] 0.5945555
```

## Assumptions to Verify

The assumptions on the above linear model are really about the joint distribution of the residuals, which are not directly observed.  On data, we try to verify:

1. The fitted values and the residuals show no trends with respect to each other
1. The residuals are distributed approximately Normal$(0, \sigma^2)$
    - A constant variance is called [**homoscedasticity**](https://en.wikipedia.org/wiki/Homoscedasticity)
    - A non-constant variance is called [**heteroscedascity**](https://en.wikipedia.org/wiki/Heteroscedasticity)
1. There are no lurking variables

There are two plots we will use in this course to investigate the first two.

## Residual Distribution


```r
> plot(myfit, which=1)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-23-1.png" width="672" style="display: block; margin: auto;" />

## Normal Residuals Check


```r
> plot(myfit, which=2)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-24-1.png" width="672" style="display: block; margin: auto;" />

##  Fitted Values Vs. Obs. Residuals

<img src="stat_models_1_files/figure-html/unnamed-chunk-25-1.png" width="960" style="display: block; margin: auto;" />


# Ordinary Least Squares


**Ordinary least squares** (OLS) estimates the model

$$
\begin{aligned}
Y_i & = \beta_1 X_{i1} + \beta_2 X_{i2} + \ldots + \beta_p X_{ip} + E_i \\
 & = \bX_i \bb + E_i
\end{aligned}
$$

where ${\rm E}[E_i] = 0$, ${\rm Var}(E_i) = \sigma^2$, and $\Cov(E_i, E_j) = 0$ for all $1 \leq i, j \leq n$ and $i \not= j$.

Note that typically $X_{i1} = 1$ for all $i$ so that $\beta_1 X_{i1} = \beta_1$ serves as the intercept.

## OLS Solution

The estimates of $\beta_1, \beta_2, \ldots, \beta_p$ are found by identifying the values that minimize:

$$ 
\begin{aligned}
\sum_{i=1}^n \left[ Y_i - (\beta_1 X_{i1} + \beta_2 X_{i2} + \ldots + \beta_p X_{ip}) \right]^2 \\
= (\bY - \bX \bb)^T (\bY - \bX \bb)
\end{aligned}
$$

The solution is expressed in terms of matrix algebra computations:

$$
\hat{\bb} = (\bX^T \bX)^{-1} \bX^T \bY.
$$

## Sample Variance

Let the predicted values of the model be

$$
\hat{\bY} = \bX \hat{\bb} = \bX (\bX^T \bX)^{-1} \bX^T \bY.
$$

We estimate $\sigma^2$ by the OLS sample variance

$$
S^2 = \frac{\sum_{i=1}^n (Y_i - \hat{Y}_i)^2}{n-p}.
$$

## Sample Covariance

The $p$-vector $\hat{\bb}$ has covariance matrix

$$
\Cov(\hat{\bb} | \bX) = (\bX^T \bX)^{-1} \sigma^2.
$$

Its estimated covariance matrix is

$$
\widehat{\Cov}(\hat{\bb}) = (\bX^T \bX)^{-1} S^2.
$$


## Expected Values

Under the assumption that ${\rm E}[E_i] = 0$, ${\rm Var}(E_i) = \sigma^2$, and $\Cov(E_i, E_j) = 0$ for all $1 \leq i, j \leq n$ and $i \not= j$, we have the following:

$$
\E\left[ \left. \hat{\bb} \right| \bX \right] = \bb
$$

$$
\E\left[ \left. S^2 \right| \bX \right] = \sigma^2
$$

$$
\E\left[\left. (\bX^T \bX)^{-1} S^2 \right| \bX\right] = \Cov\left(\hat{\bb}\right)
$$

$$
\Cov\left(\hat{\beta}_j, Y_i - \hat{Y}_i\right) = \boldsymbol{0}.
$$

## Standard Error

The standard error of $\hat{\beta}_j$ is the square root of the $(j, j)$ diagonal entry of $(\bX^T \bX)^{-1} \sigma^2$

$$
\se(\hat{\beta}_j) = \sqrt{\left[(\bX^T \bX)^{-1} \sigma^2\right]_{jj}}
$$

and estimated standard error is

$$
\hat{\se}(\hat{\beta}_j) = \sqrt{\left[(\bX^T \bX)^{-1} S^2\right]_{jj}}
$$


## Proportion of Variance Explained

The proportion of variance explained is defined equivalently to the simple linear regression scneario:

$$
R^2 = \frac{\sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2}{\sum_{i=1}^n (Y_i - \bar{Y})^2}.
$$

## Normal Errors

Suppose we assume $E_1, E_2, \ldots, E_n \iid \mbox{Normal}(0, \sigma^2)$. Then

$$
\ell\left(\bb, \sigma^2 ; \bY, \bX\right) \propto -n\log(\sigma^2) -\frac{1}{\sigma^2} (\bY - \bX \bb)^T (\bY - \bX \bb).
$$


Since minimizing $(\bY - \bX \bb)^T (\bY - \bX \bb)$ maximizes the likelihood with respect to $\bb$, this implies $\hat{\bb}$ is the MLE for $\bb$.  

It can also be calculated that $\frac{n-p}{n} S^2$ is the MLE for $\sigma^2$.

## Sampling Distribution

When $E_1, E_2, \ldots, E_n \iid \mbox{Normal}(0, \sigma^2)$, it follows that, conditional on $\bX$:

$$
\hat{\bb} \sim \mbox{MVN}_p\left(\bb, (\bX^T \bX)^{-1} \sigma^2 \right)
$$

$$
\begin{aligned}
S^2 \frac{n-p}{\sigma^2} & \sim \chi^2_{n-p} \\
\frac{\hat{\beta}_j - \beta_j}{\hat{\se}(\hat{\beta}_j)} & \sim t_{n-p}
\end{aligned}
$$

## CLT

Under the assumption that ${\rm E}[E_i] = 0$, ${\rm Var}(E_i) = \sigma^2$, and $\Cov(E_i, E_j) = 0$ for $i \not= j$, it follows that as $n \rightarrow \infty$,

$$
\sqrt{n} \left(\hat{\bb} - \bb\right) \stackrel{D}{\longrightarrow} \mbox{MVN}_p\left( \boldsymbol{0}, (\bX^T \bX)^{-1} \sigma^2 \right).
$$

## Gauss-Markov Theorem

Under the assumption that ${\rm E}[E_i] = 0$, ${\rm Var}(E_i) = \sigma^2$, and $\Cov(E_i, E_j) = 0$ for $i \not= j$, the Gauss-Markov theorem shows that among all BLUEs, **best linear unbiased estimators**, the least squares estimate has the smallest mean-squared error.

Specifically, suppose that $\tilde{\bb}$ is a linear estimator (calculated from a linear operator on $\bY$) where $\E[\tilde{\bb} | \bX] = \bb$.  Then

$$
\E\left[ \left. (\bY - \bX \hat{\bb})^T (\bY - \bX \hat{\bb}) \right| \bX \right] \leq 
\E\left[ \left. (\bY - \bX \tilde{\bb})^T (\bY - \bX \tilde{\bb}) \right| \bX \right].
$$




# Generalized Least Squares


**Generalized least squares** (GLS) assumes the same model as OLS, except it allows for **heteroskedasticity** and **covariance** among the $E_i$.  Specifically, it is assumed that $\bE = (E_1, \ldots, E_n)^T$ is distributed as

$$
\bE_{n \times 1} \sim (\boldsymbol{0}, \bSig)
$$
where $\boldsymbol{0}$ is the expected value $\bSig = (\sigma_{ij})$ is the $n \times n$ covariance matrix.


The most straightforward way to navigate GLS results is to recognize that

$$
\bSig^{-1/2} \bY = \bSig^{-1/2}\bX \bb + \bSig^{-1/2}\bE
$$

satisfies the assumptions of the OLS model.

## GLS Solution

The solution to minimizing

$$
(\bY - \bX \bb)^T \bSig^{-1} (\bY - \bX \bb)
$$

is

$$
\hat{\bb} = \left( \bX^T \bSig^{-1} \bX \right)^{-1} \bX^T \bSig^{-1} \bY.
$$

## Other Results

The issue of estimating $\bSig$ if it is unknown is complicated.  Other than estimates of $\sigma^2$, the results from the OLS section recapitulate by replacing $\bY = \bX \bb + \bE$ with 

$$
\bSig^{-1/2} \bY = \bSig^{-1/2}\bX \bb + \bSig^{-1/2}\bE.
$$



For example, as $n \rightarrow \infty$,

$$
\sqrt{n} \left(\hat{\bb} - \bb\right) \stackrel{D}{\longrightarrow} \mbox{MNV}_p\left( \boldsymbol{0}, (\bX^T \bSig^{-1} \bX)^{-1} \right).
$$

\  

We also still have that

$$
\E\left[ \left. \hat{\bb} \right| \bX \right] = \bb.
$$

\  

And when $\bE \sim \mbox{MVN}_n(\boldsymbol{0}, \bSig)$, $\hat{\bb}$ is the MLE.


# OLS in R


R implements OLS of multiple explanatory variables exactly the same as with a single explanatory variable, except we need to show the sum of all explanatory variables that we want to use.


```r
> lm(weight ~ height + sex, data=htwt)

Call:
lm(formula = weight ~ height + sex, data = htwt)

Coefficients:
(Intercept)       height         sexM  
   -76.6167       0.8106       8.2269  
```

## Weight Regressed on Height + Sex


```r
> summary(lm(weight ~ height + sex, data=htwt))

Call:
lm(formula = weight ~ height + sex, data = htwt)

Residuals:
    Min      1Q  Median      3Q     Max 
-20.131  -4.884  -0.640   5.160  41.490 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) -76.6167    15.7150  -4.875 2.23e-06 ***
height        0.8105     0.0953   8.506 4.50e-15 ***
sexM          8.2269     1.7105   4.810 3.00e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 8.066 on 197 degrees of freedom
Multiple R-squared:  0.6372,	Adjusted R-squared:  0.6335 
F-statistic:   173 on 2 and 197 DF,  p-value: < 2.2e-16
```

## One Variable, Two Scales

We can include a single variable but on two different scales:


```r
> htwt <- htwt %>% mutate(height2 = height^2)
> summary(lm(weight ~ height + height2, data=htwt))

Call:
lm(formula = weight ~ height + height2, data = htwt)

Residuals:
    Min      1Q  Median      3Q     Max 
-24.265  -5.159  -0.499   4.549  42.965 

Coefficients:
              Estimate Std. Error t value Pr(>|t|)
(Intercept) 107.117140 175.246872   0.611    0.542
height       -1.632719   2.045524  -0.798    0.426
height2       0.008111   0.005959   1.361    0.175

Residual standard error: 8.486 on 197 degrees of freedom
Multiple R-squared:  0.5983,	Adjusted R-squared:  0.5943 
F-statistic: 146.7 on 2 and 197 DF,  p-value: < 2.2e-16
```

## Interactions

It is possible to include products of explanatory variables, which is called an *interaction*.


```r
> summary(lm(weight ~ height + sex + height:sex, data=htwt))

Call:
lm(formula = weight ~ height + sex + height:sex, data = htwt)

Residuals:
    Min      1Q  Median      3Q     Max 
-20.869  -4.835  -0.897   4.429  41.122 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) -45.6730    22.1342  -2.063   0.0404 *  
height        0.6227     0.1343   4.637 6.46e-06 ***
sexM        -55.6571    32.4597  -1.715   0.0880 .  
height:sexM   0.3729     0.1892   1.971   0.0502 .  
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 8.007 on 196 degrees of freedom
Multiple R-squared:  0.6442,	Adjusted R-squared:  0.6388 
F-statistic: 118.3 on 3 and 196 DF,  p-value: < 2.2e-16
```

## More on Interactions

What happens when there is an interaction between a quantitative explanatory variable and a factor explanatory variable?  In the next plot, we show three models:

- Grey solid: `lm(weight ~ height, data=htwt)`
- Color dashed: `lm(weight ~ height + sex, data=htwt)`
- Color solid: `lm(weight ~ height + sex + height:sex, data=htwt)`

## Visualizing Three Different Models

<img src="stat_models_1_files/figure-html/unnamed-chunk-30-1.png" width="672" style="display: block; margin: auto;" />

# Categorical Explanatory Variables

## Example: Chicken Weights


```r
> data("chickwts", package="datasets")
> head(chickwts)
  weight      feed
1    179 horsebean
2    160 horsebean
3    136 horsebean
4    227 horsebean
5    217 horsebean
6    168 horsebean
> summary(chickwts$feed)
   casein horsebean   linseed  meatmeal   soybean sunflower 
       12        10        12        11        14        12 
```

## Factor Variables in `lm()`


```r
> chick_fit <- lm(weight ~ feed, data=chickwts)
> summary(chick_fit)

Call:
lm(formula = weight ~ feed, data = chickwts)

Residuals:
     Min       1Q   Median       3Q      Max 
-123.909  -34.413    1.571   38.170  103.091 

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)    323.583     15.834  20.436  < 2e-16 ***
feedhorsebean -163.383     23.485  -6.957 2.07e-09 ***
feedlinseed   -104.833     22.393  -4.682 1.49e-05 ***
feedmeatmeal   -46.674     22.896  -2.039 0.045567 *  
feedsoybean    -77.155     21.578  -3.576 0.000665 ***
feedsunflower    5.333     22.393   0.238 0.812495    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 54.85 on 65 degrees of freedom
Multiple R-squared:  0.5417,	Adjusted R-squared:  0.5064 
F-statistic: 15.36 on 5 and 65 DF,  p-value: 5.936e-10
```

## Plot the Fit


```r
> plot(chickwts$feed, chickwts$weight, xlab="Feed", ylab="Weight", las=2)
> points(chickwts$feed, chick_fit$fitted.values, col="blue", pch=20, cex=2)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-33-1.png" width="672" style="display: block; margin: auto;" />

## ANOVA (Version 1)

ANOVA (*analysis of variance*) was originally developed as a statistical model and method for comparing differences in mean values between various groups.

ANOVA quantifies and tests for differences in response variables with respect to factor variables.

In doing so, it also partitions the total variance to that due to within and between groups, where groups are defined by the factor variables.

## `anova()`

The classic ANOVA table:

```r
> anova(chick_fit)
Analysis of Variance Table

Response: weight
          Df Sum Sq Mean Sq F value    Pr(>F)    
feed       5 231129   46226  15.365 5.936e-10 ***
Residuals 65 195556    3009                      
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```


```r
> n <- length(chick_fit$residuals) # n <- 71
> (n-1)*var(chick_fit$fitted.values)
[1] 231129.2
> (n-1)*var(chick_fit$residuals)
[1] 195556
> (n-1)*var(chickwts$weight) # sum of above two quantities
[1] 426685.2
> (231129/5)/(195556/65) # F-statistic
[1] 15.36479
```

## How It Works


```r
> levels(chickwts$feed)
[1] "casein"    "horsebean" "linseed"   "meatmeal"  "soybean"   "sunflower"
> head(chickwts, n=3)
  weight      feed
1    179 horsebean
2    160 horsebean
3    136 horsebean
> tail(chickwts, n=3)
   weight   feed
69    222 casein
70    283 casein
71    332 casein
> x <- model.matrix(weight ~ feed, data=chickwts)
> dim(x)
[1] 71  6
```

## Top of Design Matrix


```r
> head(x)
  (Intercept) feedhorsebean feedlinseed feedmeatmeal feedsoybean
1           1             1           0            0           0
2           1             1           0            0           0
3           1             1           0            0           0
4           1             1           0            0           0
5           1             1           0            0           0
6           1             1           0            0           0
  feedsunflower
1             0
2             0
3             0
4             0
5             0
6             0
```

## Bottom of Design Matrix


```r
> tail(x)
   (Intercept) feedhorsebean feedlinseed feedmeatmeal feedsoybean
66           1             0           0            0           0
67           1             0           0            0           0
68           1             0           0            0           0
69           1             0           0            0           0
70           1             0           0            0           0
71           1             0           0            0           0
   feedsunflower
66             0
67             0
68             0
69             0
70             0
71             0
```

## Model Fits


```r
> chick_fit$fitted.values %>% round(digits=4) %>% unique()
[1] 160.2000 218.7500 246.4286 328.9167 276.9091 323.5833
```


```r
> chickwts %>% group_by(feed) %>% summarize(mean(weight))
# A tibble: 6 x 2
  feed      `mean(weight)`
  <fct>              <dbl>
1 casein              324.
2 horsebean           160.
3 linseed             219.
4 meatmeal            277.
5 soybean             246.
6 sunflower           329.
```

# Variable Transformations 

## Rationale

In order to obtain reliable model fits and inference on linear models, the model assumptions described earlier must be satisfied.  

Sometimes it is necessary to *transform* the response variable and/or some of the explanatory variables.

This process should involve data visualization and exploration.

## Power and Log Transformations

It is often useful to explore power and log transforms of the variables, e.g., $\log(y)$ or $y^\lambda$ for some $\lambda$ (and likewise $\log(x)$ or $x^\lambda$).

You can read more about the [Box-Cox family of power transformations](https://en.wikipedia.org/wiki/Power_transform).

## `Diamonds` Data


```r
> data("diamonds", package="ggplot2")
> head(diamonds)
# A tibble: 6 x 10
  carat cut       color clarity depth table price     x     y     z
  <dbl> <ord>     <ord> <ord>   <dbl> <dbl> <int> <dbl> <dbl> <dbl>
1 0.23  Ideal     E     SI2      61.5    55   326  3.95  3.98  2.43
2 0.21  Premium   E     SI1      59.8    61   326  3.89  3.84  2.31
3 0.23  Good      E     VS1      56.9    65   327  4.05  4.07  2.31
4 0.290 Premium   I     VS2      62.4    58   334  4.2   4.23  2.63
5 0.31  Good      J     SI2      63.3    58   335  4.34  4.35  2.75
6 0.24  Very Good J     VVS2     62.8    57   336  3.94  3.96  2.48
```

## Nonlinear Relationship


```r
> ggplot(data = diamonds) +
+   geom_point(mapping=aes(x=carat, y=price, color=clarity), alpha=0.3)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-42-1.png" width="672" style="display: block; margin: auto;" />

## Regression with Nonlinear Relationship


```r
> diam_fit <- lm(price ~ carat + clarity, data=diamonds)
> anova(diam_fit)
Analysis of Variance Table

Response: price
             Df     Sum Sq    Mean Sq  F value    Pr(>F)    
carat         1 7.2913e+11 7.2913e+11 435639.9 < 2.2e-16 ***
clarity       7 3.9082e+10 5.5831e+09   3335.8 < 2.2e-16 ***
Residuals 53931 9.0264e+10 1.6737e+06                       
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Residual Distribution


```r
> plot(diam_fit, which=1)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-44-1.png" width="672" style="display: block; margin: auto;" />

## Normal Residuals Check


```r
> plot(diam_fit, which=2)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-45-1.png" width="672" style="display: block; margin: auto;" />

## Log-Transformation


```r
> ggplot(data = diamonds) +
+   geom_point(aes(x=carat, y=price, color=clarity), alpha=0.3) +
+   scale_y_log10(breaks=c(1000,5000,10000)) + 
+   scale_x_log10(breaks=1:5)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-46-1.png" width="672" style="display: block; margin: auto;" />

## OLS on Log-Transformed Data


```r
> diamonds <- mutate(diamonds, log_price = log(price, base=10), 
+                    log_carat = log(carat, base=10))
> ldiam_fit <- lm(log_price ~ log_carat + clarity, data=diamonds)
> anova(ldiam_fit)
Analysis of Variance Table

Response: log_price
             Df Sum Sq Mean Sq   F value    Pr(>F)    
log_carat     1 9771.9  9771.9 1452922.6 < 2.2e-16 ***
clarity       7  339.1    48.4    7203.3 < 2.2e-16 ***
Residuals 53931  362.7     0.0                        
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Residual Distribution


```r
> plot(ldiam_fit, which=1)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-48-1.png" width="672" style="display: block; margin: auto;" />

## Normal Residuals Check


```r
> plot(ldiam_fit, which=2)
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-49-1.png" width="672" style="display: block; margin: auto;" />

## Tree Pollen Study



Suppose that we have a study where tree pollen measurements are averaged every week, and these data are recorded for 10 years.  These data are simulated:


```r
> pollen_study
# A tibble: 520 x 3
    week  year pollen
   <int> <int>  <dbl>
 1     1  2001  1842.
 2     2  2001  1966.
 3     3  2001  2381.
 4     4  2001  2141.
 5     5  2001  2210.
 6     6  2001  2585.
 7     7  2001  2392.
 8     8  2001  2105.
 9     9  2001  2278.
10    10  2001  2384.
# … with 510 more rows
```

## Tree Pollen Count by Week


```r
> ggplot(pollen_study) + geom_point(aes(x=week, y=pollen))
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-52-1.png" width="672" style="display: block; margin: auto;" />

## A Clever Transformation

We can see there is a linear relationship between `pollen` and `week` if we transform `week` to be number of weeks from the peak week.


```r
> pollen_study <- pollen_study %>%  
+                        mutate(week_new = abs(week-20))
```

Note that this is a very different transformation from taking a log or power transformation.

## `week` Transformed


```r
> ggplot(pollen_study) + geom_point(aes(x=week_new, y=pollen))
```

<img src="stat_models_1_files/figure-html/unnamed-chunk-54-1.png" width="672" style="display: block; margin: auto;" />


# OLS Goodness of Fit

## Pythagorean Theorem

![PythMod](images/right_triangle_model_fits.png)

Least squares model fitting can be understood through the Pythagorean theorem: $a^2 + b^2 = c^2$.  However, here we have:

$$
\sum_{i=1}^n Y_i^2 = \sum_{i=1}^n \hat{Y}_i^2 + \sum_{i=1}^n (Y_i - \hat{Y}_i)^2
$$

where the $\hat{Y}_i$ are the result of a **linear projection** of the $Y_i$.

## OLS Normal Model

In this section, let's assume that $(\bX_1, Y_1), \ldots, (\bX_n, Y_n)$ are distributed so that

$$
\begin{aligned}
Y_i & = \beta_1 X_{i1} + \beta_2 X_{i2} + \ldots + \beta_p X_{ip} + E_i \\
 & = \bX_i \bb + E_i
\end{aligned}
$$

where $\bE | \bX \sim \mbox{MVN}_n(\bO, \sigma^2 \bI)$.  Note that we haven't specified the distribution of the $\bX_i$ rv's.

## Projection Matrices

In the OLS framework we have:

$$
\hat{\bY} = \bX (\bX^T \bX)^{-1} \bX^T \bY.
$$

The matrix $\bP_{n \times n} = \bX (\bX^T \bX)^{-1} \bX^T$ is a projection matrix.  The vector $\bY$ is projected into the space spanned by the column space of $\bX$.


Project matrices have the following properties:

- $\bP$ is symmetric
- $\bP$ is idempotent so that $\bP \bP = \bP$
- If $\bX$ has column rank $p$, then $\bP$ has rank $p$
- The eigenvalues of $\bP$ are $p$ 1's and $n-p$ 0's
- The trace (sum of diagonal entries) is $\operatorname{tr}(\bP) = p$
- $\bI - \bP$ is also a projection matrix with rank $n-p$

## Decomposition

Note that $\bP (\bI - \bP) = \bP - \bP \bP = \bP - \bP = \bO$.  

We have

$$
\begin{aligned}
\| \bY\|_{2}^{2} = \bY^T \bY & = (\bP \bY + (\bI - \bP) \bY)^T (\bP \bY + (\bI - \bP) \bY) \\
 & = (\bP \bY)^T (\bP \bY) + ((\bI - \bP) \bY)^T ((\bI - \bP) \bY) \\
 & = \| \bP \bY\|_{2}^{2} + \| (\bI - \bP) \bY \|_{2}^{2}
\end{aligned}
$$

where the cross terms disappear because $\bP (\bI - \bP) = \bO$.


Note:  The $\ell_p$ norm of an $n$-vector $\boldsymbol{w}$ is defined as

$$
\| \boldsymbol{w} \|_p = \left(\sum_{i=1}^n |w_i|^p\right)^{1/p}.
$$

Above we calculated 

$$
\| \boldsymbol{w} \|_2^2 = \sum_{i=1}^n w_i^2.
$$

## Distribution of Projection

Suppose that $Y_1, Y_2, \ldots, Y_n \iid \mbox{Normal}(0,\sigma^2)$. This can also be written as $\bY \sim \mbox{MVN}_n(\bO, \sigma^2 \bI)$. It follows that 

$$
\bP \bY \sim \mbox{MVN}_{n}(\bO, \sigma^2 \bP \bI \bP^T).
$$

where $\bP \bI \bP^T = \bP \bP^T = \bP \bP = \bP$.

Also, $(\bP \bY)^T (\bP \bY) = \bY^T \bP^T \bP \bY = \bY^T \bP \bY$, a **quadratic form**.  Given the eigenvalues of $\bP$, $\bY^T \bP \bY$ is equivalent in distribution to $p$ squared iid Normal(0,1) rv's, so 

$$
\frac{\bY^T \bP \bY}{\sigma^2} \sim \chi^2_{p}.
$$

## Distribution of Residuals

If $\bP \bY = \hat{\bY}$ are the fitted OLS values, then $(\bI-\bP) \bY = \bY - \hat{\bY}$ are the residuals.

It follows by the same argument as above that

$$
\frac{\bY^T (\bI-\bP) \bY}{\sigma^2} \sim \chi^2_{n-p}.
$$

It's also straightforward to show that $(\bI-\bP)\bY \sim \mbox{MVN}_{n}(\bO, \sigma^2(\bI-\bP))$ and $\Cov(\bP\bY, (\bI-\bP)\bY) = \bO$. 


## Degrees of Freedom

The degrees of freedom, $p$, of a linear projection model fit is equal to

- The number of linearly dependent columns of $\bX$
- The number of nonzero eigenvalues of $\bP$ (where nonzero eigenvalues are equal to 1)
- The trace of the projection matrix, $\operatorname{tr}(\bP)$.

The reason why we divide estimates of variance by $n-p$ is because this is the number of effective independent sources of variation remaining after the model is fit by projecting the $n$ observations into a $p$ dimensional linear space.

## Submodels

Consider the OLS model $\bY = \bX \bb + \bE$ where there are $p$ columns of $\bX$ and $\bb$ is a $p$-vector.

Let $\bX_0$ be a subset of $p_0$ columns of $\bX$ and let $\bX_1$ be a subset of $p_1$ columns, where $1 \leq p_0 < p_1 \leq p$.  Also, assume that the columns of $\bX_0$ are a subset of $\bX_1$.

We can form $\hat{\bY}_0 = \bP_0 \bY$ where $\bP_0$ is the projection matrix built from $\bX_0$.  We can analogously form $\hat{\bY}_1 = \bP_1 \bY$.


## Hypothesis Testing

Without loss of generality, suppose that $\bb_0 = (\beta_1, \beta_2, \ldots, \beta_{p_0})^T$ and $\bb_1 = (\beta_1, \beta_2, \ldots, \beta_{p_1})^T$.

How do we compare these models, specifically to test $H_0: (\beta_{p_0+1}, \beta_{p_0 + 2}, \ldots, \beta_{p_1}) = \bO$ vs $H_1: (\beta_{p_0+1}, \beta_{p_0 + 2}, \ldots, \beta_{p_1}) \not= \bO$?

The basic idea to perform this test is to compare the goodness of fits of each model via a pivotal statistic.  We will discuss the generalized LRT and ANOVA approaches.

## Generalized LRT

Under the OLS Normal model, it follows that $\hat{\bb}_0 = (\bX^T_0 \bX_0)^{-1} \bX_0^T \bY$ is the MLE under the null hypothesis and $\hat{\bb}_1 = (\bX^T_1 \bX_1)^{-1} \bX_1^T \bY$ is the unconstrained MLE.  Also, the respective MLEs of $\sigma^2$ are

$$
\hat{\sigma}^2_0 = \frac{\sum_{i=1}^n (Y_i - \hat{Y}_{0,i})^2}{n}
$$

$$
\hat{\sigma}^2_1 = \frac{\sum_{i=1}^n (Y_i - \hat{Y}_{1,i})^2}{n}
$$

where $\hat{\bY}_{0} = \bX_0 \hat{\bb}_0$ and $\hat{\bY}_{1} = \bX_1 \hat{\bb}_1$.


The generalized LRT statistic is

$$
\lambda(\bX, \bY) = \frac{L\left(\hat{\bb}_1, \hat{\sigma}^2_1; \bX, \bY \right)}{L\left(\hat{\bb}_0, \hat{\sigma}^2_0; \bX, \bY \right)}
$$

where $2\log\lambda(\bX, \bY)$ has a $\chi^2_{p_1 - p_0}$ null distribution.

## Nested Projections

We can apply the Pythagorean theorem we saw earlier to linear subspaces to get:

$$
\begin{aligned}
\| \bY \|^2_2 & = \| (\bI - \bP_1) \bY \|_{2}^{2} + \| \bP_1 \bY\|_{2}^{2} \\
& = \| (\bI - \bP_1) \bY \|_{2}^{2} + \| (\bP_1 - \bP_0) \bY\|_{2}^{2} + \| \bP_0 \bY\|_{2}^{2}
\end{aligned}
$$

We can also use the Pythagorean theorem to decompose the residuals from the smaller projection $\bP_0$:

$$
\| (\bI - \bP_0) \bY \|^2_2 = \| (\bI - \bP_1) \bY \|^2_2 + \| (\bP_1 - \bP_0) \bY \|^2_2
$$

## *F* Statistic

The $F$ statistic compares the improvement of goodness in fit of the larger model to that of the smaller model in terms of sums of squared residuals, and it scales this improvement by an estimate of $\sigma^2$:

$$
\begin{aligned}
F & = \frac{\left[\| (\bI - \bP_0) \bY \|^2_2 - \| (\bI - \bP_1) \bY \|^2_2\right]/(p_1 - p_0)}{\| (\bI - \bP_1) \bY \|^2_2/(n-p_1)} \\
& = \frac{\left[\sum_{i=1}^n (Y_i - \hat{Y}_{0,i})^2 - \sum_{i=1}^n (Y_i - \hat{Y}_{1,i})^2 \right]/(p_1 - p_0)}{\sum_{i=1}^n (Y_i - \hat{Y}_{1,i})^2 / (n - p_1)} \\
\end{aligned}
$$


Since $\| (\bI - \bP_0) \bY \|^2_2 - \| (\bI - \bP_1) \bY \|^2_2 = \| (\bP_1 - \bP_0) \bY \|^2_2$, we can equivalently write the $F$ statistic as:

$$
\begin{aligned}
F & = \frac{\| (\bP_1 - \bP_0) \bY \|^2_2 / (p_1 - p_0)}{\| (\bI - \bP_1) \bY \|^2_2/(n-p_1)} \\
& = \frac{\sum_{i=1}^n (\hat{Y}_{1,i} - \hat{Y}_{0,i})^2 / (p_1 - p_0)}{\sum_{i=1}^n (Y_i - \hat{Y}_{1,i})^2 / (n - p_1)}
\end{aligned} 
$$

## *F* Distribution

Suppose we have independent random variables $V \sim \chi^2_a$ and $W \sim \chi^2_b$.  It follows that 

$$
\frac{V/a}{W/b} \sim F_{a,b}
$$

where $F_{a,b}$ is the $F$ distribution with $(a, b)$ degrees of freedom.


By arguments similar to those given above, we have

$$
\frac{\| (\bP_1 - \bP_0) \bY \|^2_2}{\sigma^2} \sim \chi^2_{p_1 - p_0}
$$

$$
\frac{\| (\bI - \bP_1) \bY \|^2_2}{\sigma^2} \sim \chi^2_{n-p_1}
$$

and these two rv's are independent.

## *F* Test

Suppose that the OLS model holds where $\bE | \bX \sim \mbox{MVN}_n(\bO, \sigma^2 \bI)$.

In order to test $H_0: (\beta_{p_0+1}, \beta_{p_0 + 2}, \ldots, \beta_{p_1}) = \bO$ vs $H_1: (\beta_{p_0+1}, \beta_{p_0 + 2}, \ldots, \beta_{p_1}) \not= \bO$, we can form the $F$ statistic as given above, which has null distribution $F_{p_1 - p_0, n - p_1}$.  The p-value is calculated as $\Pr(F^* \geq F)$ where $F$ is the observed $F$ statistic and $F^* \sim F_{p_1 - p_0, n - p_1}$.

If the above assumption on the distribution of $\bE | \bX$ only approximately holds, then the $F$ test p-value is also an approximation.

## Example: Davis Data


```r
> library("car")
> data("Davis", package="car")
Warning in data("Davis", package = "car"): data set 'Davis' not found
```


```r
> htwt <- tbl_df(Davis)
> htwt[12,c(2,3)] <- htwt[12,c(3,2)]
> head(htwt)
# A tibble: 6 x 5
  sex   weight height repwt repht
  <fct>  <int>  <int> <int> <int>
1 M         77    182    77   180
2 F         58    161    51   159
3 F         53    161    54   158
4 M         68    177    70   175
5 F         59    157    59   155
6 M         76    170    76   165
```

## Comparing Linear Models in R

Example: Davis Data

Suppose we are considering the three following models:


```r
> f1 <- lm(weight ~ height, data=htwt)
> f2 <- lm(weight ~ height + sex, data=htwt)
> f3 <- lm(weight ~ height + sex + height:sex, data=htwt)
```

How do we determine if the additional terms in models `f2` and `f3` are needed?

## ANOVA (Version 2)

A generalization of ANOVA exists that allows us to compare two nested models, quantifying their differences in terms of goodness of fit and performing a hypothesis test of whether this difference is statistically significant.  

A model is *nested* within another model if their difference is simply the absence of certain terms in the smaller model.  

The null hypothesis is that the additional terms have coefficients equal to zero, and the alternative hypothesis is that at least one coefficient is nonzero.

Both versions of ANOVA can be described in a single, elegant mathematical framework.

## Comparing Two Models <br> with `anova()`

This provides a comparison of the improvement in fit from model `f2` compared to model `f1`:

```r
> anova(f1, f2)
Analysis of Variance Table

Model 1: weight ~ height
Model 2: weight ~ height + sex
  Res.Df   RSS Df Sum of Sq      F    Pr(>F)    
1    198 14321                                  
2    197 12816  1    1504.9 23.133 2.999e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## When There's a Single Variable Difference  

Compare above `anova(f1, f2)` p-value to that for the `sex` term from the `f2` model:

```r
> library(broom)
> tidy(f2)
# A tibble: 3 x 5
  term        estimate std.error statistic  p.value
  <chr>          <dbl>     <dbl>     <dbl>    <dbl>
1 (Intercept)  -76.6     15.7        -4.88 2.23e- 6
2 height         0.811    0.0953      8.51 4.50e-15
3 sexM           8.23     1.71        4.81 3.00e- 6
```

## Calculating the F-statistic


```r
> anova(f1, f2)
Analysis of Variance Table

Model 1: weight ~ height
Model 2: weight ~ height + sex
  Res.Df   RSS Df Sum of Sq      F    Pr(>F)    
1    198 14321                                  
2    197 12816  1    1504.9 23.133 2.999e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

How the F-statistic is calculated:

```r
> n <- nrow(htwt)
> ss1 <- (n-1)*var(f1$residuals)
> ss1
[1] 14321.11
> ss2 <- (n-1)*var(f2$residuals)
> ss2
[1] 12816.18
> ((ss1 - ss2)/anova(f1, f2)$Df[2])/(ss2/f2$df.residual)
[1] 23.13253
```

## Calculating the Generalized LRT


```r
> anova(f1, f2, test="LRT")
Analysis of Variance Table

Model 1: weight ~ height
Model 2: weight ~ height + sex
  Res.Df   RSS Df Sum of Sq  Pr(>Chi)    
1    198 14321                           
2    197 12816  1    1504.9 1.512e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```


```r
> library(lmtest)
> lrtest(f1, f2)
Likelihood ratio test

Model 1: weight ~ height
Model 2: weight ~ height + sex
  #Df LogLik Df  Chisq Pr(>Chisq)    
1   3 -710.9                         
2   4 -699.8  1 22.205   2.45e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```


These tests produce slightly different answers because `anova()` adjusts for degrees of freedom when estimating the variance, whereas `lrtest()` is the strict generalized LRT. See [here](https://stats.stackexchange.com/questions/155474/r-why-does-lrtest-not-match-anovatest-lrt).

## ANOVA on More Distant Models

We can compare models with multiple differences in terms:


```r
> anova(f1, f3)
Analysis of Variance Table

Model 1: weight ~ height
Model 2: weight ~ height + sex + height:sex
  Res.Df   RSS Df Sum of Sq      F    Pr(>F)    
1    198 14321                                  
2    196 12567  2      1754 13.678 2.751e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Compare Multiple Models at Once

We can compare multiple models at once:


```r
> anova(f1, f2, f3)
Analysis of Variance Table

Model 1: weight ~ height
Model 2: weight ~ height + sex
Model 3: weight ~ height + sex + height:sex
  Res.Df   RSS Df Sum of Sq       F    Pr(>F)    
1    198 14321                                   
2    197 12816  1   1504.93 23.4712 2.571e-06 ***
3    196 12567  1    249.04  3.8841   0.05015 .  
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

