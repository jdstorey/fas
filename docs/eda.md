\providecommand{\E}{\operatorname{E}}
\providecommand{\V}{\operatorname{Var}}
\providecommand{\Cov}{\operatorname{Cov}}
\providecommand{\cov}{\operatorname{cov}}
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





# (PART) Expoloratory Data Analysis {-}

# Exploratory Data Analysis

## What is EDA?

Exploratory data analysis (EDA) is the process of analzying data to uncover their key features.

John Tukey pioneered this framework, writing a seminal book on the topic (called *Exploratory Data Analysis*).

EDA involves calculating numerical summaries of data, visualizing data in a variety of ways, and considering interesting data points. 

Before any model fitting is done to data, some exploratory data analysis should always be performed. 

*Data science seems to focus much more on EDA than traditional statistics.*

## Descriptive Statistics Examples

- Facebook's [Visualizing Friendships](https://www.facebook.com/note.php?note_id=469716398919)   (side note: [a discussion](http://flowingdata.com/2010/12/13/facebook-worldwide-friendships-mapped/))

- [Hans Rosling: Debunking third-world myths with the best stats you've ever seen](https://www.youtube.com/watch?v=RUwS1uAdUcI&t=3m22s&version=3)

- Flowing Data's [A Day in the Life of Americans](http://flowingdata.com/2015/12/15/a-day-in-the-life-of-americans/)

## Components of EDA

EDA involves calculating quantities and visualizing data for:

- Checking the *n*'s
- Checking for missing data
- Characterizing the distributional properties of the data
- Characterizing relationships among variables and observations
- Dimension reduction
- Model formulation
- Hypothesis generation

... and there are possible many more activities one can do.

## Data Sets

For the majority of this chapter, we will use some simple data sets to demonstrate the ideas.

### Data `mtcars`

Load the `mtcars` data set:


```r
> library("tidyverse") # why load tidyverse?
> data("mtcars", package="datasets")
> mtcars <- as_tibble(mtcars)
> head(mtcars)
# A tibble: 6 x 11
    mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
  <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
1  21       6   160   110  3.9   2.62  16.5     0     1     4     4
2  21       6   160   110  3.9   2.88  17.0     0     1     4     4
3  22.8     4   108    93  3.85  2.32  18.6     1     1     4     1
4  21.4     6   258   110  3.08  3.22  19.4     1     0     3     1
5  18.7     8   360   175  3.15  3.44  17.0     0     0     3     2
6  18.1     6   225   105  2.76  3.46  20.2     1     0     3     1
```

### Data `mpg`

Load the `mpg` data set:


```r
> data("mpg", package="ggplot2")
> head(mpg)
# A tibble: 6 x 11
  manufacturer model displ  year   cyl trans  drv     cty   hwy fl    class
  <chr>        <chr> <dbl> <int> <int> <chr>  <chr> <int> <int> <chr> <chr>
1 audi         a4      1.8  1999     4 auto(… f        18    29 p     comp…
2 audi         a4      1.8  1999     4 manua… f        21    29 p     comp…
3 audi         a4      2    2008     4 manua… f        20    31 p     comp…
4 audi         a4      2    2008     4 auto(… f        21    30 p     comp…
5 audi         a4      2.8  1999     6 auto(… f        16    26 p     comp…
6 audi         a4      2.8  1999     6 manua… f        18    26 p     comp…
```

### Data `diamonds`

Load the `diamonds` data set:


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

### Data `gapminder`

Load the `gapminder` data set:


```r
> library("gapminder")
> data("gapminder", package="gapminder")
> gapminder <- as_tibble(gapminder)
> head(gapminder)
# A tibble: 6 x 6
  country     continent  year lifeExp      pop gdpPercap
  <fct>       <fct>     <int>   <dbl>    <int>     <dbl>
1 Afghanistan Asia       1952    28.8  8425333      779.
2 Afghanistan Asia       1957    30.3  9240934      821.
3 Afghanistan Asia       1962    32.0 10267083      853.
4 Afghanistan Asia       1967    34.0 11537966      836.
5 Afghanistan Asia       1972    36.1 13079460      740.
6 Afghanistan Asia       1977    38.4 14880372      786.
```

# Numerical Summaries of Data

## Useful Summaries

- **Center**: mean, median, mode

- **Quantiles**: percentiles, five number summaries

- **Spread**: standard deviation, variance, interquartile range

- **Outliers**

- **Shape**: skewness, kurtosis

- **Concordance**: correlation, quantile-quantile plots

## Measures of Center

Suppose we have data points $x_1, x_2, \ldots, x_n$.  

**Mean**: $$\overline{x} = \frac{x_1 + x_2 + \cdots + x_n}{n}$$

**Median**: Order the points $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$.  The median is the middle value:  
- $x_{((n+1)/2)}$ if $n$ is odd  
- $(x_{(n/2)} + x_{(n/2+1)})/2$ if $n$ is even

**Mode**: The most frequently repeated value among the data (if any).  If there are ties, then there is more than one mode.

## Mean, Median, and Mode in R

Let's calculate these quantities in R.


```r
> mean(mtcars$mpg)
[1] 20.09062
> median(mtcars$mpg)
[1] 19.2
> 
> sample_mode <- function(x) {
+   as.numeric(names(which(table(x) == max(table(x)))))
+ }
> 
> sample_mode(round(mtcars$mpg))
[1] 15 21
```

It appears there is no R base function for calculating the mode.

## Quantiles and Percentiles

The $p$th **percentile** of $x_1, x_2, \ldots, x_n$ is a number such that $p$% of the data are less than this number.

The 25th, 50th, and 75th percentiles are called 1st, 2nd, and 3rd "quartiles", respectively. These are sometimes denoted as Q1, Q2, and Q3. The median is the 50th percentile aka the 2nd quartile aka Q2.

In general, $q$-**quantiles** are cut points that divide the data into $q$ approximately equally sized groups.  The cut points are the percentiles $1/q, 2/q, \ldots, (q-1)/q.$

## Five Number Summary

The "five number summary" is the minimum, the three quartiles, and the maximum. This can be calculated via `fivenum()` and `summary()`. [They can produce different values.](https://chemicalstatistician.wordpress.com/2013/08/12/exploratory-data-analysis-the-5-number-summary-two-different-methods-in-r-2/) Finally, `quantile()` extracts any set of percentiles.


```r
> fivenum(mtcars$mpg)
[1] 10.40 15.35 19.20 22.80 33.90
> summary(mtcars$mpg)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  10.40   15.43   19.20   20.09   22.80   33.90 
> 
> quantile(mtcars$mpg, prob=seq(0, 1, 0.25))
    0%    25%    50%    75%   100% 
10.400 15.425 19.200 22.800 33.900 
```

## Measures of Spread

The variance, standard deviation (SD), and interquartile range (IQR) measure the "spread" of the data.

**Variance**:
$$s^2 = \frac{\sum_{i=1}^n \left(x_i - \overline{x}\right)^2}{n-1}$$

**Standard Deviation**: $s = \sqrt{s^2}$

**Iterquartile Range**: IQR $=$ Q3 $-$ Q1 

The SD and IQR have the same units as the observed data, but the variance is in squared units.


## Variance, SD, and IQR in R

Variance:

```r
> var(mtcars$mpg)
[1] 36.3241
```

Standard deviation:

```r
> sd(mtcars$mpg)
[1] 6.026948
```

Interquartile range:

```r
> IQR(mtcars$mpg)
[1] 7.375
> diff(fivenum(mtcars$mpg)[c(2,4)])
[1] 7.45
```

## Identifying Outliers

An **outlier** is an unusual data point.  Outliers can be perfectly valid but they can also be due to errors (as can non-outliers). 

One must define what is meant by an outlier. 

One definition is a data point that less than Q1 or greater than Q3 by 1.5 $\times$ IQR or more.

Another definition is a data point whose difference from the mean is greater than 3 $\times$ SD or more. For Normal distributed data (bell curve shaped), the probability of this is less than 0.27%.

## Application to `mtcars` Data


```r
> sd_units <- abs(mtcars$wt - mean(mtcars$wt))/sd(mtcars$wt)
> sum(sd_units > 3)
[1] 0
> max(sd_units)
[1] 2.255336
> 
> iqr_outlier_cuts <- fivenum(mtcars$wt)[c(2,4)] + 
+       c(-1.5, 1.5)*diff(fivenum(mtcars$wt)[c(2,4)])
> sum(mtcars$wt < iqr_outlier_cuts[1] | 
+     mtcars$wt > iqr_outlier_cuts[2])
[1] 2
```

## Measuring Symmetry

The **skewness** statistic measures symmetry of the data.  It is calculated by:

$$
\gamma = \frac{\sum_{i=1}^n (x_i - \overline{x})^3/n}{s^3}
$$

A negative number is left-skewed, and a positive number is right-skewed.  

Note:  Use of $n$ vs. $n-1$ may vary -- check the code.

## `skewness()` Function

In R, there is a function call `skewness()` from the `moments` package for calculating this statistic on data.


```r
> library(moments)
> gapminder %>% filter(year==2007) %>% select(gdpPercap) %>% 
+   skewness()
gdpPercap 
 1.211228 
> gapminder %>% filter(year==2007) %>% select(gdpPercap) %>% 
+   log() %>% skewness()
 gdpPercap 
-0.1524203 
> rnorm(10000) %>% skewness()
[1] 0.005799917
```

## Measuring Tails

The tails of a distribution are often described as being heavy or light depending on how slowly they descend.

This can be measured through statistic called **kurtosis**:

$$
\kappa = \frac{\sum_{i=1}^n (x_i - \overline{x})^4/n}{s^4}
$$
As with skewness $\gamma$, use of $n$ vs $n-1$ may vary.

## Excess Kurtosis

For a standard Normal distribution (mean 0 and standard deviation 1), the kurtosis is on average 3.

Therefore, a measure called "excess kurtosis" is defined to be $\kappa - 3$.  A positive value implies heavier tails and a negative value implies lighter tails.

## `kurtosis()` Function

In R, there is a function call `kurtosis()` from the `moments` package for calculating this statistic on data.


```r
> library(moments)
> gapminder %>% filter(year==2007) %>% select(gdpPercap) %>% 
+   kurtosis()
gdpPercap 
  3.29593 
> gapminder %>% filter(year==2007) %>% select(gdpPercap) %>% 
+   log() %>% kurtosis()
gdpPercap 
 1.871608 
> rnorm(10000) %>% kurtosis()
[1] 2.955853
```

## Visualizing Skewness and Kurtosis

<img src="eda_files/figure-html/unnamed-chunk-15-1.png" width="672" style="display: block; margin: auto;" />

<img src="eda_files/figure-html/unnamed-chunk-16-1.png" width="672" style="display: block; margin: auto;" />

## Covariance and Correlation

- It is often the case that two or more quantitative variables are measured on each unit of observation (such as an individual).  

- We are then often interested in characterizing how pairs of variables are associated or how they vary together.

- Two common measures for this are called "covariance" and "correlation", both of which are most well suited for measuring linear associations

### Covariance

Suppose we observe $n$ pairs of data $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$. Their sample covariance is

$$
\cov_{xy} = \frac{\sum_{i=1}^n (x_i - \overline{x}) (y_i - \overline{y})}{(n-1)},
$$
which meausers how the two variables "covary" about their respective means.  Large positive numbers indicate concordance of deviations from the mean, and large negative numbers indicated discordance (so opposite sides of the mean).

### Pearson Correlation

Pearson correlation is sample covariance scaled by the variables' standard deviations, meaning correlation is a unitless measure of variation about the mean. It is defined by 

\begin{eqnarray}
r_{xy} & = & \frac{\sum_{i=1}^n (x_i - \overline{x}) (y_i - \overline{y})}{\sqrt{\sum_{i=1}^n (x_i - \overline{x})^2 \sum_{i=1}^n (y_i - \overline{y})^2}} \\
\ & = & \frac{\sum_{i=1}^n (x_i - \overline{x}) (y_i - \overline{y})}{(n-1) s_x s_y} \\
\ & = & \frac{\cov_{xy}}{s_x s_y}
\end{eqnarray}

where $s_x$ and $s_y$ are the sample standard deviations of each measured variable. Note that $-1 \leq r_{xy} \leq 1$.

### Spearman Correlation

There are other ways to measure correlation that are less reliant on linear trends in covariation and are also more robust to outliers. Specifically, one can convert each measured variable to ranks by size (1 for the smallest, $n$ for the largest) and then use a formula for correlation designed for these ranks. One popular measure of rank-based correlation is the [Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).





```r
> x <- rnorm(500)
> y <- x + rnorm(500)
> cor(x, y, method="pearson")
[1] 0.7542651
> cor(x, y, method="spearman")
[1] 0.7499555
```

<img src="eda_files/figure-html/unnamed-chunk-19-1.png" width="672" style="display: block; margin: auto;" />





```r
> x <- rnorm(500)
> y <- x + rnorm(500, sd=2)
> cor(x, y, method="pearson")
[1] 0.5164903
> cor(x, y, method="spearman")
[1] 0.5093092
```

<img src="eda_files/figure-html/unnamed-chunk-22-1.png" width="672" style="display: block; margin: auto;" />





```r
> x <- c(rnorm(499), 100)
> y <- c(rnorm(499), 100)
> cor(x, y, method="pearson")
[1] 0.9528564
> cor(x, y, method="spearman")
[1] -0.02133551
```

<img src="eda_files/figure-html/unnamed-chunk-25-1.png" width="960" style="display: block; margin: auto;" />


# Data Visualization Basics

## Plots

- Single variables:
    - Barplot
    - Boxplot
    - Histogram
    - Density plot
- Two or more variables:
    - Side-by-Side Boxplots
    - Stacked Barplot
    - Scatterplot

## R Base Graphics

- We'll first plodding through "R base graphics", which means graphics functions that come with R.  
- By default they are very simple.  However, they can be customized *a lot*, but it takes *a lot* of work. 
- Also, the syntax varies significantly among plot types and some think the syntax is not user-friendly.
- We will consider a very highly used graphics package next week, called `ggplot2` that provides a "grammar of graphics". It hits a sweet spot of "flexibility vs. complexity" for many data scientists.

## Read the Documentation

For all of the plotting functions covered below, read the help files.


```r
> ?barplot
> ?boxplot
> ?hist
> ?density
> ?plot
> ?legend
```

## Barplot


```r
> cyl_tbl <- table(mtcars$cyl)
> barplot(cyl_tbl, xlab="Cylinders", ylab="Count")
```

<img src="eda_files/figure-html/unnamed-chunk-28-1.png" width="672" style="display: block; margin: auto;" />

## Boxplot


```r
> boxplot(mtcars$mpg, ylab="MPG", col="lightgray")
```

<img src="eda_files/figure-html/unnamed-chunk-29-1.png" width="672" style="display: block; margin: auto;" />

## Constructing Boxplots

- The top of the box is Q3
- The line through the middle of the box is the median
- The bottom of the box is Q1
- The top whisker is the minimum of Q3 + 1.5 $\times$ IQR or the largest data point
- The bottom whisker is the maximum of Q1 - 1.5 $\times$ IQR or the smallest data point
- Outliers lie outside of (Q1 - 1.5 $\times$ IQR) or (Q3 + 1.5 $\times$ IQR), and they are shown as points
- Outliers are calculated using the `fivenum()` function


## Boxplot with Outliers


```r
> boxplot(mtcars$wt, ylab="Weight (1000 lbs)", 
+         col="lightgray")
```

<img src="eda_files/figure-html/unnamed-chunk-30-1.png" width="672" style="display: block; margin: auto;" />

## Histogram


```r
> hist(mtcars$mpg, xlab="MPG", main="", col="lightgray")
```

<img src="eda_files/figure-html/unnamed-chunk-31-1.png" width="672" style="display: block; margin: auto;" />

## Histogram with More Breaks


```r
> hist(mtcars$mpg, breaks=12, xlab="MPG", main="", col="lightgray")
```

<img src="eda_files/figure-html/unnamed-chunk-32-1.png" width="672" style="display: block; margin: auto;" />


## Density Plot


```r
> plot(density(mtcars$mpg), xlab="MPG", main="")
> polygon(density(mtcars$mpg), col="lightgray", border="black")
```

<img src="eda_files/figure-html/unnamed-chunk-33-1.png" width="672" style="display: block; margin: auto;" />

## Boxplot (Side-By-Side)


```r
> boxplot(mpg ~ cyl, data=mtcars, xlab="Cylinders", 
+         ylab="MPG", col="lightgray")
```

<img src="eda_files/figure-html/unnamed-chunk-34-1.png" width="672" style="display: block; margin: auto;" />

## Stacked Barplot


```r
> counts <- table(mtcars$cyl, mtcars$gear)
> counts
   
     3  4  5
  4  1  8  2
  6  2  4  1
  8 12  0  2
```


```r
> barplot(counts, main="Number of Gears and Cylinders",
+   xlab="Gears", col=c("blue","red", "lightgray"))
> legend(x="topright", title="Cyl",
+        legend = rownames(counts), 
+        fill = c("blue","red", "lightgray"))
```


<img src="eda_files/figure-html/unnamed-chunk-37-1.png" width="672" style="display: block; margin: auto;" />



## Scatterplot


```r
> plot(mtcars$wt, mtcars$mpg, xlab="Weight (1000 lbs)", 
+      ylab="MPG")
```

<img src="eda_files/figure-html/unnamed-chunk-38-1.png" width="672" style="display: block; margin: auto;" />

## Quantile-Quantile Plots

Quantile-quantile plots display the [quantiles](#/quantiles-and-percentiles) of: 

1. two samples of data
2. a sample of data vs a theoretical distribution

The first type allows one to assess how similar the distributions are of two samples of data.

The second allows one to assess how similar a sample of data is to a theoretical distribution (often Normal with mean 0 and standard deviation 1).



```r
> qqnorm(mtcars$mpg, main=" ")
> qqline(mtcars$mpg) # line through Q1 and Q3
```

<img src="eda_files/figure-html/unnamed-chunk-39-1.png" width="672" style="display: block; margin: auto;" />

  

```r
> before1980 <- gapminder %>% filter(year < 1980) %>% 
+   select(lifeExp) %>% unlist()
> after1980 <- gapminder %>% filter(year > 1980) %>% 
+   select(lifeExp) %>% unlist()
> qqplot(before1980, after1980); abline(0,1)
```

<img src="eda_files/figure-html/unnamed-chunk-40-1.png" width="672" style="display: block; margin: auto;" />



```r
> ggplot(mtcars) + stat_qq(aes(sample = mpg))
```

<img src="eda_files/figure-html/unnamed-chunk-41-1.png" width="672" style="display: block; margin: auto;" />



```r
> ggplot(gapminder) + stat_qq(aes(sample=lifeExp))
```

<img src="eda_files/figure-html/unnamed-chunk-42-1.png" width="672" style="display: block; margin: auto;" />



```r
> ggplot(gapminder) + 
+   stat_qq(aes(sample=lifeExp, color=continent))
```

<img src="eda_files/figure-html/unnamed-chunk-43-1.png" width="672" style="display: block; margin: auto;" />


# A Grammar of Graphics

## Rationale

A grammar for communicating data visualization:

- *Data*: the data set we are plotting
- *Aesthetics*: the variation or relationships in the data we want to visualize
- *Geometries*: the geometric object by which we render the aesthetics
- *Coordinates*: the coordinate system used (not covered here)
- *Facets*: the layout of plots required to visualize the data
- Other Options:  any other customizations we wish to make, such as changing the color scheme or labels

These are strung together like words in a sentence.

## Package `ggplot2` 

The R package `ggplot2` implements a grammar of graphics along these lines.  First, let's load `ggplot2`:


```r
> library(ggplot2)
```

Now let's set a theme (more on this later):


```r
> theme_set(theme_bw())
```

## Pieces of the Grammar

- `ggplot()`
- `aes()`
- `geom_*()`
- `facet_*()`
- `scale_*()`
- `theme()`
- `labs()`

The `*` is a placeholder for a variety of terms that we will consider.

## Geometries

Perhaps the most important aspect of `ggplot2` is to understand the "geoms". We will cover the following:

- `geom_bar()`
- `geom_boxplot()`
- `geom_violin()`
- `geom_histogram()`
- `geom_density()`
- `geom_line()`
- `geom_point()`
- `geom_smooth()`
- `geom_hex()`

## Call Format

The most basic `ggplot2` plot is made with something like:

```
ggplot(data = <DATA FRAME>) +
 geom_*(mapping = aes(x = <VAR X>, y = <VAR Y>))
``` 

where `<DATA FRAME>` is a data frame and `<VAR X>` and `<VAR Y>` are variables (i.e., columns) from this data frame.  Recall `geom_*` is a placeholder for a geometry such as `geom_boxplot`.

## Layers

There's a complex "layers" construct occurring in the `ggplot2` package. However, for our purposes, it suffices to note that the different parts of the plots are layered together through the `+` operator:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy, color=drv)) +
+   geom_smooth(mapping = aes(x = displ, y = hwy, color=drv)) + 
+   scale_color_brewer(palette = "Set1", name = "Drivetrain") +
+   labs(title = "Highway MPG By Drivetrain and Displacement", 
+        x = "Displacement", y = "Highway MPG")
```

## Placement of the `aes()` Call

In the previous slide, we saw that the same `aes()` call was made for two `geom`'s.  When this is the case, we may more simply call `aes()` from within `ggplot()`:


```r
> ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color=drv)) + 
+   geom_point() +
+   geom_smooth() + 
+   scale_color_brewer(palette = "Set1", name = "Drivetrain") +
+   labs(title = "Highway MPG By Drivetrain and Displacement", 
+        x = "Displacement", y = "Highway MPG")
```

There may be cases where different `geom`'s are layered and require different `aes()` calls.  This is something to keep in mind as we go through the specifics of the `ggplot2` package.

## Original Publications

Wickham, H. (2010) [A Layered Grammar of Graphics.](http://www.tandfonline.com/doi/abs/10.1198/jcgs.2009.07098) *Journal of Computational and Graphical Statistics*, 19 (1): 3--28.

This paper designs an implementation of *The Grammar of Graphics* by Leland Wilkinson (published in 2005).

## Documentation

- In R: `help(package="ggplot2")`
- <http://docs.ggplot2.org/current/>
- <http://www.cookbook-r.com/Graphs/>
- [*ggplot2: Elegant Graphics for Data Analysis*](http://amzn.com/0387981403) (somewhat outdated, but gives clear rationale)

## Barplots


The `geom_bar()` layer forms a barplot and only requires an `x` assignment in the `aes()` call:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut))
```

<img src="eda_files/figure-html/unnamed-chunk-48-1.png" width="672" style="display: block; margin: auto;" />


Color in the bars by assigning `fill` in `geom_bar()`, but outside of `aes()`:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut), fill = "tomato")
```

<img src="eda_files/figure-html/unnamed-chunk-49-1.png" width="672" style="display: block; margin: auto;" />



Color *within* the bars according to a variable by assigning `fill` in `geom_bar()` *inside* of `aes()`:



```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = cut))
```

<img src="eda_files/figure-html/unnamed-chunk-50-1.png" width="672" style="display: block; margin: auto;" />


When we use `fill = clarity` within `aes()`, we see that it shows the proportion of each `clarity` value within each `cut` value:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = clarity))
```

<img src="eda_files/figure-html/unnamed-chunk-51-1.png" width="672" style="display: block; margin: auto;" />


By setting `position = "dodge"` outside of `aes()`, it shows bar charts for the `clarity` values within each `cut` value:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping= aes(x = cut, fill = clarity), 
+            position = "dodge")
```

<img src="eda_files/figure-html/unnamed-chunk-52-1.png" width="672" style="display: block; margin: auto;" />


By setting `position = "fill"`, it shows the proportion of `clarity` values within each `cut` value and no longer shows the `cut` values:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping=aes(x = cut, fill = clarity), 
+            position = "fill") +
+   labs(x = "cut", y = "relative proporition within cut")
```

<img src="eda_files/figure-html/unnamed-chunk-53-1.png" width="672" style="display: block; margin: auto;" />

## Boxplots and Violin Plots


The `geom_boxplot()` layer forms a boxplot and requires both `x` and `y` assignments in the `aes()` call, even when plotting a single boxplot:


```r
> ggplot(data = mpg) + 
+   geom_boxplot(mapping = aes(x = 1, y = hwy))
```

<img src="eda_files/figure-html/unnamed-chunk-54-1.png" width="672" style="display: block; margin: auto;" />


Color in the boxes by assigning `fill` in `geom_boxplot()`, but outside of `aes()`:


```r
> ggplot(data = mpg) + 
+   geom_boxplot(mapping = aes(x = 1, y = hwy), 
+                fill="lightblue") +
+   labs(x=NULL)
```

<img src="eda_files/figure-html/unnamed-chunk-55-1.png" width="672" style="display: block; margin: auto;" />


Show a boxplot for the `y` values occurring within each `x` factor level by making these assignments in `aes()`:


```r
> ggplot(data = mpg) + 
+   geom_boxplot(mapping = aes(x = factor(cyl), y = hwy))
```

<img src="eda_files/figure-html/unnamed-chunk-56-1.png" width="672" style="display: block; margin: auto;" />


By assigning the `fill` argument *within* `aes()`, we can color each boxplot according to the x-axis factor variable:


```r
> ggplot(data = mpg) + 
+   geom_boxplot(mapping = aes(x = factor(cyl), y = hwy, 
+                              fill = factor(cyl)))
```

<img src="eda_files/figure-html/unnamed-chunk-57-1.png" width="672" style="display: block; margin: auto;" />


The `geom_jitter()` function plots the data points and randomly jitters them so we can better see all of the points:


```r
> ggplot(data = mpg, mapping = aes(x=factor(cyl), y=hwy)) + 
+   geom_boxplot(fill = "lightblue") +
+   geom_jitter(width = 0.2)
```

<img src="eda_files/figure-html/unnamed-chunk-58-1.png" width="672" style="display: block; margin: auto;" />


A violin plot, called via `geom_violin()`, is similar to a boxplot, except shows a density plot turned on its side and reflected across its vertical axis:


```r
> ggplot(data = mpg) + 
+   geom_violin(mapping = aes(x = drv, y = hwy))
```

<img src="eda_files/figure-html/unnamed-chunk-59-1.png" width="672" style="display: block; margin: auto;" />


Add a `geom_jitter()` to see how the original data points relate to the violin plots:


```r
> ggplot(data = mpg, mapping = aes(x = drv, y = hwy)) + 
+   geom_violin(adjust=1.2) +
+   geom_jitter(width=0.2, alpha=0.5)
```

<img src="eda_files/figure-html/unnamed-chunk-60-1.png" width="672" style="display: block; margin: auto;" />


Boxplot example on the `gapminder` data:


```r
> ggplot(gapminder, aes(x = continent, y = lifeExp)) +
+   geom_boxplot(outlier.colour = "red") +
+   geom_jitter(width = 0.1, alpha = 0.25)
```

<img src="eda_files/figure-html/unnamed-chunk-61-1.png" width="672" style="display: block; margin: auto;" />


Analogous violin plot example on the `gapminder` data:


```r
> ggplot(gapminder, aes(x = continent, y = lifeExp)) +
+   geom_violin() +
+   geom_jitter(width = 0.1, alpha = 0.25)
```

<img src="eda_files/figure-html/unnamed-chunk-62-1.png" width="672" style="display: block; margin: auto;" />


## Histograms and Density Plots


We can create a histogram using the `geom_histogram()` layer, which requires an `x` argument only in the `aes()` call:


```r
> ggplot(gapminder) +
+   geom_histogram(mapping = aes(x=lifeExp))
```

<img src="eda_files/figure-html/unnamed-chunk-63-1.png" width="672" style="display: block; margin: auto;" />


We can change the bin width directly in the histogram, which is an intuitive parameter to change based on visual inspection:


```r
> ggplot(gapminder) +
+   geom_histogram(mapping = aes(x=lifeExp), binwidth=5)
```

<img src="eda_files/figure-html/unnamed-chunk-64-1.png" width="672" style="display: block; margin: auto;" />


The bins are sometimes centered in an unexpected manner in `ggplot2`:


```r
> ggplot(diamonds) +
+   geom_histogram(mapping = aes(x=price), binwidth = 1000)
```

<img src="eda_files/figure-html/unnamed-chunk-65-1.png" width="672" style="display: block; margin: auto;" />


Let's fix how the bins are centered (make `center` half of `binwidth`).


```r
> ggplot(diamonds) +
+   geom_histogram(mapping = aes(x=price), binwidth = 1000, 
+                  center=500)
```

<img src="eda_files/figure-html/unnamed-chunk-66-1.png" width="672" style="display: block; margin: auto;" />


Instead of counts on the y-axis, we may instead want the area of the bars to sum to 1, like a probability density:


```r
> ggplot(gapminder) +
+   geom_histogram(mapping = aes(x=lifeExp, y=..density..), 
+                  binwidth=5)
```

<img src="eda_files/figure-html/unnamed-chunk-67-1.png" width="672" style="display: block; margin: auto;" />


When we use `fill = continent` within `aes()`, we see that it shows the counts of each `continent` value within each `lifeExp` bin:


```r
> ggplot(gapminder) +
+   geom_histogram(mapping = aes(x=lifeExp, fill = continent), 
+                  binwidth = 5)
```

<img src="eda_files/figure-html/unnamed-chunk-68-1.png" width="672" style="display: block; margin: auto;" />


Display a density plot using the `geom_density()` layer:


```r
> ggplot(gapminder) +
+   geom_density(mapping = aes(x=lifeExp))
```

<img src="eda_files/figure-html/unnamed-chunk-69-1.png" width="672" style="display: block; margin: auto;" />


Employ the arguments `color="blue"` and `fill="lightblue"` outside of the `aes()` call to include some colors:


```r
> ggplot(gapminder) +
+   geom_density(mapping = aes(x=lifeExp), color="blue", 
+                fill="lightblue")
```

<img src="eda_files/figure-html/unnamed-chunk-70-1.png" width="672" style="display: block; margin: auto;" />


By utilizing `color=as.factor(year)` we plot a density of `lifeExp` stratified by each `year` value:


```r
> ggplot(gapminder) +
+   geom_density(aes(x=lifeExp, color=as.factor(year)), 
+                size=1.2)
```

<img src="eda_files/figure-html/unnamed-chunk-71-1.png" width="672" style="display: block; margin: auto;" />


Overlay a density plot and a histogram together:


```r
> ggplot(gapminder, mapping = aes(x=lifeExp)) + 
+   geom_histogram(aes(y=..density..), color="black", 
+                  fill="white") +
+   geom_density(fill="lightblue", alpha=.5)
```

<img src="eda_files/figure-html/unnamed-chunk-72-1.png" width="672" style="display: block; margin: auto;" />

## Line Plots



<h2>`babynames` Revisited</h2>

Let's first create a data frame that captures the number of times "John" is registered in males per year:


```r
> library("babynames")
> john <- babynames %>% filter(sex=="M", name=="John")
> head(john)
# A tibble: 6 x 5
   year sex   name      n   prop
  <dbl> <chr> <chr> <int>  <dbl>
1  1880 M     John   9655 0.0815
2  1881 M     John   8769 0.0810
3  1882 M     John   9557 0.0783
4  1883 M     John   8894 0.0791
5  1884 M     John   9388 0.0765
6  1885 M     John   8756 0.0755
```


We can `geom_lines()` to plot a line showing the popularity of "John" over time:


```r
> ggplot(data = john) + 
+   geom_line(mapping = aes(x=year, y=prop), size=1.5)
```

<img src="eda_files/figure-html/unnamed-chunk-74-1.png" width="672" style="display: block; margin: auto;" />


Now let's look at a name that occurs nontrivially in males and females:


```r
> kelly <- babynames %>% filter(name=="Kelly")
> ggplot(data = kelly) + 
+   geom_line(mapping = aes(x=year, y=prop, color=sex), 
+             size=1.5)
```

<img src="eda_files/figure-html/unnamed-chunk-75-1.png" width="672" style="display: block; margin: auto;" />

## Scatterplots


The layer `geom_point()` produces a scatterplot, and the `aes()` call requires `x` and `y` assignment:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy))
```

<img src="eda_files/figure-html/unnamed-chunk-76-1.png" width="672" style="display: block; margin: auto;" />


Give the points a color:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy), 
+              color = "blue")
```

<img src="eda_files/figure-html/unnamed-chunk-77-1.png" width="672" style="display: block; margin: auto;" />


Color the points according to a factor variable by including `color = class` within the `aes()` call:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy, 
+                            color = class))
```

<img src="eda_files/figure-html/unnamed-chunk-78-1.png" width="672" style="display: block; margin: auto;" />


Increase the size of points with `size=2` outside of the `aes()` call:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy, 
+                            color = class), size=2)
```

<img src="eda_files/figure-html/unnamed-chunk-79-1.png" width="672" style="display: block; margin: auto;" />


Vary the size of the points according to the `pop` variable:


```r
> gapminder %>% filter(year==2007) %>% ggplot() + 
+   geom_point(aes(x = log(gdpPercap), y = lifeExp, 
+                  size = pop))
```

<img src="eda_files/figure-html/unnamed-chunk-80-1.png" width="672" style="display: block; margin: auto;" />


Vary the transparency of the points according to the `class` factor variable by setting `alpha=class` within the `aes()` call:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy, 
+                            alpha = class))
Warning: Using alpha for a discrete variable is not advised.
```

<img src="eda_files/figure-html/unnamed-chunk-81-1.png" width="672" style="display: block; margin: auto;" />


Vary the shape of the points according to the `class` factor variable by setting `alpha=class` within the `aes()` call (maximum 6 possible shapes -- oops!):


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy, 
+                            shape = class))
```

<img src="eda_files/figure-html/unnamed-chunk-82-1.png" width="672" style="display: block; margin: auto;" />


Color the points according to the `cut` variable by setting `color=cut` within the `aes()` call:


```r
> ggplot(data = diamonds) +
+   geom_point(mapping = aes(x=carat, y=price, color=cut),
+              alpha=0.7)
```

<img src="eda_files/figure-html/unnamed-chunk-83-1.png" width="672" style="display: block; margin: auto;" />


Color the points according to the `clarity` variable by setting `color=clarity` within the `aes()` call:


```r
> ggplot(data = diamonds) +
+   geom_point(mapping=aes(x=carat, y=price, color=clarity), 
+              alpha=0.3)
```

<img src="eda_files/figure-html/unnamed-chunk-84-1.png" width="672" style="display: block; margin: auto;" />



Override the `alpha=0.3` in the legend:


```r
> ggplot(data=diamonds) +
+   geom_point(mapping=aes(x=carat, y=price, color=clarity), 
+              alpha=0.3) + 
+   guides(color=guide_legend(override.aes = list(alpha = 1)))
```

<img src="eda_files/figure-html/unnamed-chunk-85-1.png" width="672" style="display: block; margin: auto;" />

## Axis Scales


A different way to take the log of `gdpPercap`:


```r
> gapminder %>% filter(year==2007) %>% ggplot() + 
+   geom_point(aes(x = gdpPercap, y = lifeExp, 
+                  size = pop)) +
+   scale_x_log10()
```

<img src="eda_files/figure-html/unnamed-chunk-86-1.png" width="672" style="display: block; margin: auto;" />


The `price` variable seems to be significantly right-skewed:


```r
> ggplot(diamonds) + 
+   geom_boxplot(aes(x=color, y=price)) 
```

<img src="eda_files/figure-html/unnamed-chunk-87-1.png" width="672" style="display: block; margin: auto;" />


We can try to reduce this skewness by rescaling the variables.  We first try to take the `log(base=10)` of the `price` variable via `scale_y_log10()`:


```r
> ggplot(diamonds) + 
+   geom_boxplot(aes(x=color, y=price)) + 
+   scale_y_log10()
```

<img src="eda_files/figure-html/unnamed-chunk-88-1.png" width="672" style="display: block; margin: auto;" />


Let's repeat this on the analogous violing plots:


```r
> ggplot(diamonds) + 
+   geom_violin(aes(x=color, y=price)) + 
+   scale_y_log10()
```

<img src="eda_files/figure-html/unnamed-chunk-89-1.png" width="672" style="display: block; margin: auto;" />


The relationship between `carat` and `price` is nonlinear.  Let's explore different transformations to find an approximately linear relationship.


```r
> ggplot(data = diamonds) +
+   geom_point(mapping=aes(x=carat, y=price, color=clarity), 
+              alpha=0.3)
```

<img src="eda_files/figure-html/unnamed-chunk-90-1.png" width="672" style="display: block; margin: auto;" />


First try to take the squareroot of the the `price` variable:


```r
> ggplot(data = diamonds) +
+   geom_point(aes(x=carat, y=price, color=clarity), 
+              alpha=0.3) +
+   scale_y_sqrt()
```

<img src="eda_files/figure-html/unnamed-chunk-91-1.png" width="672" style="display: block; margin: auto;" />


Now let's try to take `log(base=10)` on both the `carat` and `price` variables:


```r
> ggplot(data = diamonds) +
+   geom_point(aes(x=carat, y=price, color=clarity), alpha=0.3) +
+   scale_y_log10(breaks=c(1000,5000,10000)) + 
+   scale_x_log10(breaks=1:5)
```

<img src="eda_files/figure-html/unnamed-chunk-92-1.png" width="672" style="display: block; margin: auto;" />


Forming a violin plot of `price` stratified by `clarity` and transforming the `price` variable yields an interesting relationship in this data set:


```r
> ggplot(diamonds) + 
+   geom_violin(aes(x=clarity, y=price, fill=clarity), 
+               adjust=1.5) +  
+   scale_y_log10()
```

<img src="eda_files/figure-html/unnamed-chunk-93-1.png" width="672" style="display: block; margin: auto;" />

## Scatterplot Smoothers


<h2>Fitting "Smoothers" and Other Models to Scatterplots</h2>

- Later this semester, we will spend several weeks learning how to explain or predict an outcome variable in terms of predictor variables  
- We will briefly show here how to plot some simple model fits to scatterplots
- You may want to return to these slides later in the semester once we cover modeling in more detail


Recall the scatterplot showing the relationship between highway mpg and displacement.  How can we plot a smoothed relationship between these two variables?


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy))
```

<img src="eda_files/figure-html/unnamed-chunk-94-1.png" width="672" style="display: block; margin: auto;" />


Plot a smoother with `geom_smooth()` using the default settings (other than removing the error bands):


```r
> ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
+   geom_point() + 
+   geom_smooth(se=FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-95-1.png" width="672" style="display: block; margin: auto;" />


The default smoother here is a "loess" smoother. Let's compare that to the least squares regresson line:


```r
> ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) + 
+   geom_point() + 
+   geom_smooth(aes(colour = "loess"), method = "loess", se = FALSE) + 
+   geom_smooth(aes(colour = "lm"), method = "lm", se = FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-96-1.png" width="672" style="display: block; margin: auto;" />


Now let's plot a smoother to the points stratified by the `drv` variable:


```r
> ggplot(data=mpg, mapping = aes(x = displ, y = hwy, 
+                                linetype = drv)) + 
+   geom_point() + 
+   geom_smooth(se=FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-97-1.png" width="672" style="display: block; margin: auto;" />


Instead of different line types, let's instead differentiate them by line color:


```r
> ggplot(data = mpg, mapping = aes(x = displ, y = hwy, 
+                                  color=drv)) + 
+   geom_point() +
+   geom_smooth(se=FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-98-1.png" width="672" style="display: block; margin: auto;" />

## Overplotting


<h2> Definition </h2>

- Overplotting occurs when there are many observations, resulting in many objects being plotted on top of each other
- For example, the `diamonds` data set has 53940 observations per variable
- Let's explore some ways to deal with overplotting


Here is an example of an overplotted scatterplot:


```r
> ggplot(data = diamonds, mapping = aes(x=carat, y=price)) +
+   geom_point()
```

<img src="eda_files/figure-html/unnamed-chunk-99-1.png" width="672" style="display: block; margin: auto;" />


Let's reduce the `alpha` of the points:


```r
> ggplot(data = diamonds, mapping = aes(x=carat, y=price)) +
+   geom_point(alpha=0.1)
```

<img src="eda_files/figure-html/unnamed-chunk-100-1.png" width="672" style="display: block; margin: auto;" />


Let's further reduce the `alpha`:


```r
> ggplot(data = diamonds, mapping = aes(x=carat, y=price)) +
+   geom_point(alpha=0.01)
```

<img src="eda_files/figure-html/unnamed-chunk-101-1.png" width="672" style="display: block; margin: auto;" />


We can bin the points into hexagons, and report how many points fall within each bin.  We use the `geom_hex()` layer to do this:


```r
> ggplot(data = diamonds, mapping = aes(x=carat, y=price)) +
+   geom_hex()
```

<img src="eda_files/figure-html/unnamed-chunk-102-1.png" width="672" style="display: block; margin: auto;" />


Let's try to improve the color scheme:


```r
> ggplot(data = diamonds, mapping = aes(x=carat, y=price)) +
+   geom_hex() + 
+   scale_fill_gradient2(low="lightblue", mid="purple", high="black", 
+                        midpoint=3000)
```

<img src="eda_files/figure-html/unnamed-chunk-103-1.png" width="672" style="display: block; margin: auto;" />


We can combine the scale transformation used earlier with the "hexbin" plotting method:


```r
> ggplot(data = diamonds, mapping = aes(x=carat, y=price)) +
+   geom_hex(bins=20) +
+   scale_x_log10(breaks=1:5) + scale_y_log10(breaks=c(1000,5000,10000)) 
```

<img src="eda_files/figure-html/unnamed-chunk-104-1.png" width="672" style="display: block; margin: auto;" />


## Labels and Legends


Here's how you can change the axis labels and give the plot a title:


```r
> ggplot(data = mpg) + 
+   geom_boxplot(mapping = aes(x = factor(cyl), y = hwy)) +
+   labs(title="Highway MPG by Cylinders",x="Cylinders",
+        y="Highway MPG")
```

<img src="eda_files/figure-html/unnamed-chunk-105-1.png" width="672" style="display: block; margin: auto;" />


You can remove the legend to a plot by the following:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = cut)) +
+   theme(legend.position="none")
```

<img src="eda_files/figure-html/unnamed-chunk-106-1.png" width="672" style="display: block; margin: auto;" />


The legend can be placed on the "top", "bottom", "left", or "right":


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = cut)) +
+   theme(legend.position="bottom")
```

<img src="eda_files/figure-html/unnamed-chunk-107-1.png" width="672" style="display: block; margin: auto;" />


The legend can be moved inside the plot itself:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = cut)) +
+   theme(legend.position=c(0.15,0.75))
```

<img src="eda_files/figure-html/unnamed-chunk-108-1.png" width="672" style="display: block; margin: auto;" />


Change the name of the legend:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = cut)) +
+   scale_fill_discrete(name="Diamond\nCut")
```

<img src="eda_files/figure-html/unnamed-chunk-109-1.png" width="672" style="display: block; margin: auto;" />


Change the labels within the legend:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = cut)) +
+   scale_fill_discrete(labels=c("F", "G", "VG", "P", "I"))
```

<img src="eda_files/figure-html/unnamed-chunk-110-1.png" width="672" style="display: block; margin: auto;" />

## Facets


Here is the histogram of the `displ` variable from the `mpg` data set:


```r
> ggplot(mpg) + geom_histogram(mapping=aes(x=displ), 
+                              binwidth=0.25)
```

<img src="eda_files/figure-html/unnamed-chunk-111-1.png" width="672" style="display: block; margin: auto;" />


The `facet_wrap()` layer allows us to stratify the `displ` variable according to `cyl`, and show the histograms for the strata in an organized fashion:


```r
> ggplot(mpg) + 
+   geom_histogram(mapping=aes(x=displ), binwidth=0.25) + 
+   facet_wrap(~ cyl)
```

<img src="eda_files/figure-html/unnamed-chunk-112-1.png" width="672" style="display: block; margin: auto;" />


Here is `facet_wrap()` applied to `displ` startified by the `drv` variable:


```r
> ggplot(mpg) + 
+   geom_histogram(mapping=aes(x=displ), binwidth=0.25) + 
+   facet_wrap(~ drv)
```

<img src="eda_files/figure-html/unnamed-chunk-113-1.png" width="672" style="display: block; margin: auto;" />


We can stratify by two variable simultaneously by using the `facet_grid()` layer:


```r
> ggplot(mpg) + 
+   geom_histogram(mapping=aes(x=displ), binwidth=0.25) + 
+   facet_grid(drv ~ cyl)
```

<img src="eda_files/figure-html/unnamed-chunk-114-1.png" width="672" style="display: block; margin: auto;" />


Let's carry out a similar faceting on the `diamonds` data over the next four plots:


```r
> ggplot(diamonds) + 
+   geom_histogram(mapping=aes(x=price), binwidth=500)
```

<img src="eda_files/figure-html/unnamed-chunk-115-1.png" width="672" style="display: block; margin: auto;" />


Stratify `price` by `clarity`:


```r
> ggplot(diamonds) + 
+   geom_histogram(mapping=aes(x=price), binwidth=500) + 
+   facet_wrap(~ clarity)
```

<img src="eda_files/figure-html/unnamed-chunk-116-1.png" width="672" style="display: block; margin: auto;" />


Stratify `price` by `clarity`, but allow each y-axis range to be different by including the `scale="free_y"` argument:


```r
> ggplot(diamonds) + 
+   geom_histogram(mapping=aes(x=price), binwidth=500) + 
+   facet_wrap(~ clarity, scale="free_y")
```

<img src="eda_files/figure-html/unnamed-chunk-117-1.png" width="672" style="display: block; margin: auto;" />


Jointly stratify `price` by `cut` and `clarify`:


```r
> ggplot(diamonds) + 
+   geom_histogram(mapping=aes(x=price), binwidth=500) + 
+   facet_grid(cut ~ clarity) +
+   scale_x_continuous(breaks=9000)
```

<img src="eda_files/figure-html/unnamed-chunk-118-1.png" width="672" style="display: block; margin: auto;" />

## Colors

### Finding Colors

- [A list](http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf) of named colors in R (e.g., "lightblue")
- [RColorBrewer](https://cran.r-project.org/web/packages/RColorBrewer/index.html) package
- The Crayola crayon colors from the [`broman`](https://cran.r-project.org/web/packages/broman/index.html) package -- use `brocolors(set="crayons")`
- [Color blind palette](http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/): 


```r
> cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", 
+                "#D55E00", "#CC79A7")
```

### Some Useful Layers

- `scale_fill_manual()`
- `scale_color_manual()`
- `scale_fill_gradient()`
- `scale_color_gradient()`



Manually determine colors to fill the barplot using the color blind palette defined above, `cbPalette`:


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut, fill = cut)) +
+   scale_fill_manual(values=cbPalette)
```

<img src="eda_files/figure-html/unnamed-chunk-120-1.png" width="672" style="display: block; margin: auto;" />



Manually determine point colors using the color blind palette defined above, `cbPalette`:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy, color = class), size=2) +
+   scale_color_manual(values=cbPalette)
```

<img src="eda_files/figure-html/unnamed-chunk-121-1.png" width="672" style="display: block; margin: auto;" />


Fill the histogram bars using a color gradient by their counts, where we determine the endpoint colors:


```r
> ggplot(data = mpg) + 
+   geom_histogram(aes(x=hwy, fill=..count..)) + 
+   scale_fill_gradient(low="blue", high="red")
```

<img src="eda_files/figure-html/unnamed-chunk-122-1.png" width="672" style="display: block; margin: auto;" />


Color the points based on a gradient formed from the quantitative variable, `displ`, where we we determine the endpoint colors:


```r
> ggplot(data = mpg) + 
+   geom_point(aes(x=hwy, y=cty, color=displ), size=2) + 
+   scale_color_gradient(low="blue", high="red")
```

<img src="eda_files/figure-html/unnamed-chunk-123-1.png" width="672" style="display: block; margin: auto;" />


An example of using the palette "Set1" from the `RColorBrewer` package, included in `ggplot2`:


```r
> ggplot(diamonds) +
+   geom_density(mapping = aes(x=price, color=clarity)) +
+   scale_color_brewer(palette = "Set1")
```

<img src="eda_files/figure-html/unnamed-chunk-124-1.png" width="672" style="display: block; margin: auto;" />


Another example of using the palette "Set1" from the `RColorBrewer` package, included in `ggplot2`:


```r
> ggplot(data = mpg) + 
+   geom_point(mapping = aes(x = displ, y = hwy, color = class)) +
+   scale_color_brewer(palette = "Set1")
```

<img src="eda_files/figure-html/unnamed-chunk-125-1.png" width="672" style="display: block; margin: auto;" />


The `gapminder` package comes with its own set of colors, `country_colors`.


```r
> ggplot(subset(gapminder, continent != "Oceania"),
+        aes(x = year, y = lifeExp, group = country, 
+            color = country)) +
+   geom_line(show.legend = FALSE) + facet_wrap(~ continent) +
+   scale_color_manual(values = country_colors)
```

<img src="eda_files/figure-html/unnamed-chunk-126-1.png" width="672" style="display: block; margin: auto;" />

## Saving Plots

### Saving Plots as Variables

Pieces of the plots can be saved as variables, which is particular useful to explortatory data analysis. These all produce the same plot:


```r
> ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color=drv)) + 
+   geom_point() +
+   geom_smooth(se=FALSE)
```


```r
> p <- ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color=drv)) +  
+   geom_point() 
> p + geom_smooth(se=FALSE)
```


```r
> p <- ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color=drv)) 
> p +  geom_point() + geom_smooth(se=FALSE)
```

Try it yourself!

### Saving Plots to Files

Plots can be saved to many formats using the `ggsave()` function.  Here are some examples:


```r
> p <- ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color=drv)) + 
+   geom_point() +
+   geom_smooth(se=FALSE)
> ggsave(filename="my_plot.pdf", plot=p) # saves PDF file
> ggsave(filename="my_plot.png", plot=p) # saves PNG file
```

Here are the arguments that `ggsave()` takes:

```r
> str(ggsave)
function (filename, plot = last_plot(), device = NULL, path = NULL, 
    scale = 1, width = NA, height = NA, units = c("in", "cm", "mm"), 
    dpi = 300, limitsize = TRUE, ...)  
```

## Dynamic Visualization

### Examples

Tools to dynamically interact with data visualizations (and calculations) are becoming increasingly common and straightforward to implement.  Here are several examples:

- [Shiny](https://shiny.rstudio.com)  (see also, [example from my lab](http://qvalue.princeton.edu))
- [plotly](https://plot.ly/r/)
- [ggvis](http://ggvis.rstudio.com)
- [animation](https://cran.r-project.org/web/packages/animation/index.html)
- [gganimate](https://github.com/dgrtwo/gganimate)



```r
> p <- ggplot(gapminder) +
+         geom_point(aes(x=gdpPercap, y=lifeExp, size = pop, 
+                    color = continent, frame = year)) +
+         scale_x_log10()
> gganimate(p, "animation_ex1.gif", ani.height=400, ani.width=500)
```

The resulting file can be viewed here: [https://github.com/jdstorey/asdslectures/blob/master/docs/images/animation_ex1.gif](https://raw.githubusercontent.com/jdstorey/asdslectures/master/docs/images/animation_ex1.gif)



```r
> p <- ggplot(gapminder) +
+   geom_density(aes(x=lifeExp, color=as.factor(year), 
+                    frame=year), 
+                size=1.2) +
+   scale_color_discrete(name="year")
> gganimate(p, "animation_ex2.gif", ani.height=400, ani.width=500)
```

The resulting file can be viewed here: [https://github.com/jdstorey/asdslectures/blob/master/docs/images/animation_ex2.gif](https://raw.githubusercontent.com/jdstorey/asdslectures/master/docs/images/animation_ex2.gif)




```r
> p <- ggplot(gapminder) +
+   geom_density(aes(x=lifeExp, color=as.factor(year), 
+                    frame=year, cumulative = TRUE), 
+                size=1.2) +
+   scale_color_discrete(name="year")
> gganimate(p, "animation_ex3.gif", ani.height=400, ani.width=500)
```

The resulting file can be viewed here: [https://github.com/jdstorey/asdslectures/blob/master/docs/images/animation_ex3.gif](https://raw.githubusercontent.com/jdstorey/asdslectures/master/docs/images/animation_ex3.gif)


## Themes

### Available Themes

See <https://r4ds.had.co.nz/graphics-for-communication.html#themes> for an explanation of the `ggplot2` themes.  

See also the [`ggthemes`](https://cran.r-project.org/web/packages/ggthemes/index.html) package for additional themes.

### Setting a Theme

Globally:


```r
> theme_set(theme_minimal())
```

Locally: 


```r
> ggplot(data = diamonds) + 
+   geom_bar(mapping = aes(x = cut)) + 
+   theme_minimal()
```

# EDA of High-Dimensional Data

## Definition

**High-dimensional data** (HD data) typically refers to data sets where *many variables* are simultaneously measured on any number of observations.

The number of variables is often represented by $p$ and the number of observations by $n$.  

HD data are collected into a $p \times n$ or $n \times p$ matrix.  

Many methods exist for "large $p$, small $n$" data sets.

## Examples

- Clinical studies
- Genomics (e.g., gene expression)
- Neuroimaging (e.g., fMRI)
- Finance (e.g., time series)
- Environmental studies
- Internet data (e.g., Netflix movie ratings)

## Big Data vs HD Data

"Big data" are data sets that cannot fit into a standard computer's memory.

HD data were defined above.

They are not necessarily equivalent.

## Definition of HD Data

**High-dimesional data** is a data set where the number of variables measured is many.

**Large same size** data is a data set where few variables are measured, but many observations are measured.

**Big data** is a data set where there are so many data points that it cannot be managed straightforwardly in memory, but must rather be stored and accessed elsewhere.  Big data can be high-dimensional, large sample size, or both.

We will abbreviate high-dimensional with **HD**.

## Rationale

Exploratory data analysis (EDA) of high-dimensional data adds the additional challenge that many variables must be examined simultaneously.  Therefore, in addition to the EDA methods we discussed earlier, methods are often employed to organize, visualize, or numerically capture high-dimensional data into lower dimensions.

Examples of EDA approaches applied to HD data include:

- Traditional EDA methods covered earlier
- Cluster analysis
- Dimensionality reduction

# Cluster Analysis

## Definition

**Cluster analysis** is the process of grouping objects (variables or observations) into groups based on measures of similarity.  

Similar objects are placed in the same cluster, and dissimilar objects are placed in different clusters.

Cluster analysis methods are typically described by algorithms (rather than models or formulas).

## Types of Clustering

Clustering can be categorized in various ways:

- Hard vs. soft
- Top-down vs bottom-up
- Partitioning vs. hierarchical agglomerative

## Top-Down vs Bottom-Up

We will discuss two of the major clustering methods -- *hierarchical clustering* and *K-means clustering*.

Hierarchical clustering is an example of *bottom-up* clustering in that the process begings with each object being its own cluster and then objects are joined in a hierarchical manner into larger and larger clusters.

$K$-means clustering is an example of *top-down* clustering in that the number of clusters is chosen beforehand and then object are assigned to one of the $K$ clusters.

## Challenges {#clustering-challenges}

- Cluster analysis method
- Distance measure
- Number of clusters
- Convergence issues

## Illustrative Data Sets

### Simulated `data1`

<img src="eda_files/figure-html/unnamed-chunk-134-1.png" width="672" style="display: block; margin: auto;" />

### "True" Clusters `data1`

<img src="eda_files/figure-html/unnamed-chunk-135-1.png" width="672" style="display: block; margin: auto;" />

### Simulated `data2`

<img src="eda_files/figure-html/unnamed-chunk-136-1.png" width="672" style="display: block; margin: auto;" />

### "True" Clusters `data2`

<img src="eda_files/figure-html/unnamed-chunk-137-1.png" width="672" style="display: block; margin: auto;" />

## Distance Measures

### Objects

Most clustering methods require calculating a "distance" between two objects.

Let $\pmb{a} = (a_1, a_2, \ldots, a_n)$ be one object and $\pmb{b} = (b_1, b_2, \ldots, b_n)$ be another object.

We will assume both objects are composed of real numbers.

### Euclidean

Euclidean distance is the shortest spatial distance between two objects in Euclidean space.

Euclidean distance is calculated as:

$$d(\pmb{a}, \pmb{b}) = \sqrt{\sum_{i=1}^n \left(a_i - b_i \right)^2}$$

### Manhattan

Manhattan distance is sometimes called taxicab distance.  If you picture two locations in a city, it is the distance a taxicab must travel to get from one location to the other.

Manhattan distance is calculated as:

$$d(\pmb{a}, \pmb{b}) = \sum_{i=1}^n \left| a_i - b_i \right|$$

### Euclidean vs Manhattan

<center>![distances](images/distances.jpg)</center>

<font size=3em>
Green is Euclidean.  All others are Manhattan (and equal). Figure from [*Exploratory Data Analysis with R*](https://leanpub.com/exdata).
</font>

### `dist()`

A distance matrix -- which is the set of values resulting from a distance measure applied to all pairs of objects -- can be obtained through the function `dist()`.

Default arguments for `dist()`:

```r
> str(dist)
function (x, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)  
```

The key argument for us is `method=` which can take values `method="euclidean"` and `method="manhattan"` among others.  See `?dist`.

### Distance Matrix `data1`


```r
> sub_data1 <- data1[1:4, c(1,2)]
> sub_data1
         x        y
1 2.085818 2.248086
2 1.896636 1.369547
3 2.097729 2.386383
4 1.491026 2.029814
> mydist <- dist(sub_data1)
> print(mydist)
          1         2         3
2 0.8986772                    
3 0.1388086 1.0365293          
4 0.6335776 0.7749019 0.7037257
```


```r
> (sub_data1[1,] - sub_data1[2,])^2 %>% sum() %>% sqrt()
[1] 0.8986772
```

## Hierarchical Clustering

### Strategy

Hierarchical clustering is a hierarchical agglomerative, bottom-up clustering method that strategically joins objects into larger and larger clusters, until all objects are contained in a single cluster.

Hierarchical clustering results are typically displayed as a [dendrogram](https://en.wikipedia.org/wiki/Dendrogram).

The number of clusters does not necessarily need to be known or chosen by the analyst.

### Example: Cancer Subtypes

<center>![cancer_clustering](images/cancer_clustering.jpg)</center>

<font size=3em>
Figure from [Alizadeh et al. (2000) *Nature*](http://www.nature.com/nature/journal/v403/n6769/abs/403503a0.html).
</font>

### Algorithm

The algorithm for hierarchical clustering works as follows.

1. Start with each object assigned as its own cluster.
2. Calculate a distance between all pairs of clusters.
3. Join the two clusters with the smallest distance.
4. Repeat steps 2--3 until there is only one cluster.

At the very first iteration of the algorithm, all we need is some distance function (e.g., Euclidean or Manhattan) to determine the two objects that are closest.  But once clusters with more than one object are present, how do we calculate the distance between two clusters?  This is where a key choice called the *linkage method or criterion* is needed.

### Linkage Criteria

Suppose there are two clusters $A$ and $B$ and we have a distance function $d(\pmb{a}, \pmb{b})$ for all objects $\pmb{a} \in A$ and $\pmb{b} \in B$.  Here are three ways (among many) to calculate a distance between clusters $A$ and $B$:

\begin{eqnarray}
\mbox{Complete: } & \max \{d(\pmb{a}, \pmb{b}): \pmb{a} \in A, \pmb{b} \in B\} \\
\mbox{Single: } & \min \{d(\pmb{a}, \pmb{b}): \pmb{a} \in A, \pmb{b} \in B\} \\
\mbox{Average: } & \frac{1}{|A| |B|} \sum_{\pmb{a} \in A} \sum_{\pmb{b} \in B} d(\pmb{a}, \pmb{b})
\end{eqnarray}

### `hclust()`

The `hclust()` function produces an R object that contains all of the information needed to create a complete hierarchical clustering.

Default arguments for `hclust()`:

```r
> str(hclust)
function (d, method = "complete", members = NULL)  
```

The primary input for `hclust()` is the `d` argument, which is a distance matrix (usually obtained from `dist()`).  The `method` argument takes the linkage method, which includes `method="complete"`, `method="single"`, `method="average"`, etc.  See `?hclust`.

### Hierarchical Clustering of `data1`

<img src="eda_files/figure-html/unnamed-chunk-142-1.png" width="672" style="display: block; margin: auto;" />

### Standard `hclust()` Usage


```r
> mydist <- dist(data1, method = "euclidean")
> myhclust <- hclust(mydist, method="complete")
> plot(myhclust)
```

<img src="eda_files/figure-html/unnamed-chunk-143-1.png" width="672" style="display: block; margin: auto;" />

### `as.dendrogram()`


```r
> plot(as.dendrogram(myhclust))
```

<img src="eda_files/figure-html/unnamed-chunk-144-1.png" width="672" style="display: block; margin: auto;" />

### Modify the Labels


```r
> library(dendextend)
> dend1 <- as.dendrogram(myhclust)
> labels(dend1) <- data1$true_clusters
> labels_colors(dend1) <- 
+   c("red", "blue", "gray47")[as.numeric(data1$true_clusters)]
> plot(dend1, axes=FALSE, main=" ", xlab=" ")
```

<img src="eda_files/figure-html/unnamed-chunk-145-1.png" width="672" style="display: block; margin: auto;" />

### Color the Branches


```r
> dend2 <- as.dendrogram(myhclust)
> labels(dend2) <- rep(" ", nrow(data1))
> dend2 <- color_branches(dend2, k = 3, col=c("red", "blue", "gray47"))
> plot(dend2, axes=FALSE, main=" ", xlab=" ")
```

<img src="eda_files/figure-html/unnamed-chunk-146-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 3$)


```r
> est_clusters <- cutree(myhclust, k=3)
> est_clusters
  [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [36] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 [71] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3
[106] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
[141] 3 3 3 3 3 3 3 3 3 3
```


```r
> est_clusters <- factor(est_clusters)
> p <- data1 %>% 
+   mutate(est_clusters=est_clusters) %>% 
+   ggplot()
> p + geom_point(aes(x=x, y=y, color=est_clusters))
```

### Cluster Assignments ($K = 3$)

<img src="eda_files/figure-html/unnamed-chunk-149-1.png" width="672" style="display: block; margin: auto;" />


### Cluster Assignments ($K = 2$)


```r
> (data1 %>% 
+    mutate(est_clusters=factor(cutree(myhclust, k=2))) %>% 
+    ggplot()) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-150-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 4$)


```r
> (data1 %>% 
+    mutate(est_clusters=factor(cutree(myhclust, k=4))) %>% 
+    ggplot()) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-151-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 6$)


```r
> (data1 %>% 
+    mutate(est_clusters=factor(cutree(myhclust, k=6))) %>% 
+    ggplot()) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-152-1.png" width="672" style="display: block; margin: auto;" />

### Linkage: Complete (Default)


```r
> data1 %>% dist() %>% hclust(method="complete") %>% 
+   as.dendrogram() %>% plot(axes=FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-153-1.png" width="672" style="display: block; margin: auto;" />

### Linkage: Average


```r
> data1 %>% dist() %>% hclust(method="average") %>% 
+   as.dendrogram() %>% plot(axes=FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-154-1.png" width="672" style="display: block; margin: auto;" />

### Linkage: Single


```r
> data1 %>% dist() %>% hclust(method="single") %>% 
+   as.dendrogram() %>% plot(axes=FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-155-1.png" width="672" style="display: block; margin: auto;" />

### Linkage: Ward


```r
> data1 %>% dist() %>% hclust(method="ward.D") %>% 
+   as.dendrogram() %>% plot(axes=FALSE)
```

<img src="eda_files/figure-html/unnamed-chunk-156-1.png" width="672" style="display: block; margin: auto;" />

### Hierarchical Clustering of `data2`

<img src="eda_files/figure-html/unnamed-chunk-157-1.png" width="672" style="display: block; margin: auto;" />

### `as.dendrogram()`


```r
> mydist <- dist(data2, method = "euclidean")
> myhclust <- hclust(mydist, method="complete")
> plot(as.dendrogram(myhclust))
```

<img src="eda_files/figure-html/unnamed-chunk-158-1.png" width="672" style="display: block; margin: auto;" />

### Modify the Labels


```r
> library(dendextend)
> dend1 <- as.dendrogram(myhclust)
> labels(dend1) <- data2$true_clusters
> labels_colors(dend1) <- 
+   c("red", "blue")[as.numeric(data2$true_clusters)]
> plot(dend1, axes=FALSE, main=" ", xlab=" ")
```

<img src="eda_files/figure-html/unnamed-chunk-159-1.png" width="672" style="display: block; margin: auto;" />

### Color the Branches


```r
> dend2 <- as.dendrogram(myhclust)
> labels(dend2) <- rep(" ", nrow(data2))
> dend2 <- color_branches(dend2, k = 2, col=c("red", "blue"))
> plot(dend2, axes=FALSE, main=" ", xlab=" ")
```

<img src="eda_files/figure-html/unnamed-chunk-160-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 2$)


```r
> (data2 %>% 
+    mutate(est_clusters=factor(cutree(myhclust, k=2))) %>% 
+    ggplot()) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-161-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 3$)


```r
> (data2 %>% 
+    mutate(est_clusters=factor(cutree(myhclust, k=3))) %>% 
+    ggplot()) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-162-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 4$)


```r
> (data2 %>% 
+    mutate(est_clusters=factor(cutree(myhclust, k=4))) %>% 
+    ggplot()) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-163-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 5$)


```r
> (data2 %>% 
+    mutate(est_clusters=factor(cutree(myhclust, k=6))) %>% 
+    ggplot()) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-164-1.png" width="672" style="display: block; margin: auto;" />


## K-Means Clustering

### Strategy

K-means clustering is a top-down, partitioning cluster analysis method that assigns each object to one of $K$ clusters based on the distance between each object and the cluster centers, called *centroids*.

This is an iterative algorithm with potential random initial values.

The value of $K$ is typically unknown and must be determined by the analyst.

### Centroid

A centroid is the coordinate-wise average of all objects in a cluster.

Let $A$ be a given cluster with objects $\pmb{a} \in A$. Its centroid is:

$$\overline{\pmb{a}} = \frac{1}{|A|} \sum_{\pmb{a} \in A} \pmb{a}$$ 

### Algorithm

The number of clusters $K$ must be chosen beforehand.

1. Initialize $K$ cluster centroids.
2. Assign each object to a cluster by choosing the cluster with the smalllest distance (e.g., Euclidean) between the object and the cluster centroid.
3. Calculate new centroids based on the cluster assignments from Step 2.
4. Repeat Steps 2--3 until convergence.

### Notes

The initialization of the centroids is typically random, so often the algorithm is run several times with new, random initial centroids.

Convergence is usually defined in terms of neglible changes in the centroids or no changes in the cluster assignments.  

### `kmeans()`

K-means clustering can be accomplished through the following function:

```r
> str(kmeans)
function (x, centers, iter.max = 10L, nstart = 1L, algorithm = c("Hartigan-Wong", 
    "Lloyd", "Forgy", "MacQueen"), trace = FALSE)  
```

- `x`: the data to clusters, objects along rows
- `centers`: either the number of clusters $K$ or a matrix giving initial centroids
- `iter.max`: the maximum number of iterations allowed
- `nstart`: how many random intial $K$ centroids, where the best one is returned

### `fitted()`

The cluster centroids or assigments can be extracted through the function `fitted()`, which is applied to the output of `kmeans()`.  

The input of `fitted()` is the object returned by `kmeans()`.  The key additional argument is called `method`.

When `method="centers"` it returns the centroids.  When `method="classes"` it returns the cluster assignments.

### K-Means Clustering of `data1`


```r
> km1 <- kmeans(x=data1[,-3], centers=3, iter.max=100, nstart=5)
> est_clusters <- fitted(km1, method="classes")
> est_clusters
  [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [36] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 [71] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2
[106] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
[141] 2 2 2 2 2 2 2 2 2 2
```

### Centroids of `data1`


```r
> centroids1 <- fitted(km1, method="centers") %>% unique()
> centroids1
         x        y
1 1.943184 2.028062
3 2.042872 4.037987
2 4.015934 2.962279
```

```r
> est_clusters <- fitted(km1, method="classes")
> data1 %>% mutate(est_clusters = factor(est_clusters)) %>% 
+   group_by(est_clusters) %>% summarize(mean(x), mean(y))
# A tibble: 3 x 3
  est_clusters `mean(x)` `mean(y)`
  <fct>            <dbl>     <dbl>
1 1                 1.94      2.03
2 2                 4.02      2.96
3 3                 2.04      4.04
```

### Cluster Assignments ($K = 3$)


```r
> est_clusters <- factor(est_clusters)
> ggplot(data1) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-169-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 2$)

<img src="eda_files/figure-html/unnamed-chunk-170-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 6$)

<img src="eda_files/figure-html/unnamed-chunk-171-1.png" width="672" style="display: block; margin: auto;" />

### K-Means Clustering of `data2`


```r
> km2 <- kmeans(x=data2[,-3], centers=2, iter.max=100, nstart=5)
> est_clusters <- fitted(km2, method="classes")
> est_clusters
  [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 [36] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2
 [71] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
[106] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
```

### Cluster Assignments ($K = 2$)


```r
> est_clusters <- factor(est_clusters)
> ggplot(data2) + geom_point(aes(x=x, y=y, color=est_clusters))
```

<img src="eda_files/figure-html/unnamed-chunk-173-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 3$)

<img src="eda_files/figure-html/unnamed-chunk-174-1.png" width="672" style="display: block; margin: auto;" />

### Cluster Assignments ($K = 5$)

<img src="eda_files/figure-html/unnamed-chunk-175-1.png" width="672" style="display: block; margin: auto;" />



# Principal Component Analysis

## Dimensionality Reduction

The goal of **dimensionality reduction** is to extract low dimensional representations of high dimensional data that are useful for visualization, exploration, inference, or prediction.

The low dimensional representations should capture key sources of variation in the data.

Some methods for dimensionality reduction include:

- Principal component analysis
- Singular value decomposition 
- Latent variable modeling
- Vector quantization
- Self-organizing maps
- Multidimensional scaling

We will focus on what is likely the most commonly applied dimensionality reduction tool, principal components analysis.

## Goal of PCA

For a given set of variables, **principal component analysis** (PCA) finds (constrained) weighted sums of the variables to produce variables (called principal components) that capture consectuive maximum levels of variation in the data.

Specifically, the first principal component is the weighted sum of the variables that results in a component with the highest variation.  

This component is then "removed" from the data, and the second principal component is obtained on the resulting residuals.  

This process is repeated until there is no variation left in the data.

## Defining the First PC

Suppose we have $m$ variables, each with $n$ observations:


$$
\begin{aligned}
\bx_1 & = (x_{11}, x_{12}, \ldots, x_{1n}) \\
\bx_2 & = (x_{21}, x_{22}, \ldots, x_{2n}) \\
\  & \vdots \ \\
\bx_m & = (x_{m1}, x_{m2}, \ldots, x_{mn})
\end{aligned}
$$

We can organize these variables into an $m \times n$ matrix $\bX$ where row $i$ is $\bx_i$.

Consider all possible weighted sums of these variables

$$\tilde{\pmb{x}} = \sum_{i=1}^{m} u_i \pmb{x}_i$$

where we constrain $\sum_{i=1}^{m} u_i^2 = 1$. We wish to identify the vector $\pmb{u} = \{u_i\}_{i=1}^{m}$ under this constraint that maximizes the sample variance of $\tilde{\pmb{x}}$. However, note that if we first mean center each variable, replacing $x_{ij}$ with 
$$x_{ij}^* = x_{ij} -  \frac{1}{n} \sum_{k=1}^n x_{ik},$$ 
then the sample variance of $\tilde{\pmb{x}} = \sum_{i=1}^{m} u_i \pmb{x}_i$ is equal to that of $\tilde{\pmb{x}}^* = \sum_{i=1}^{m} u_i \pmb{x}^*_i$. 

PCA is a method concerned with decompositon variance and covariance, so we don't wish to involve the mean of each indivdual variable. Therefore, unless the true population mean of each variable is known (in which case it would be subtracted from its respective variable), we will formulate PCA in terms of mean centered variables, $\pmb{x}^*_i =  (x^*_{i1}, x^*_{i2}, \ldots, x^*_{in})$, which we can collect into $m \times n$ matrix $\bX^*$. We therefore consider all possible weighted sums of variables:
$$\tilde{\pmb{x}}^* = \sum_{i=1}^{m} u_i \pmb{x}^*_i.$$


The **first principal component** of $\bX^*$ (and $\bX$) is $\tilde{\pmb{x}}^*$ with maximum sample variance

$$
s^2_{\tilde{\bx}^*}  = \frac{\sum_{j=1}^n \tilde{x}^{*2}_j}{n-1} 
$$

The $\pmb{u} = \{u_i\}_{i=1}^{m}$ yielding this first principal component is called its **loadings**.

Note that 
$$
s^2_{\tilde{\bx}}  = \frac{\sum_{j=1}^n \left(\tilde{x}_j - \frac{1}{n} \sum_{k=1}^n \tilde{x}_k \right)^2}{n-1} = \frac{\sum_{j=1}^n \tilde{x}^{*2}_j}{n-1} = s^2_{\tilde{\bx}^*} \ ,
$$
so the loadings can be found from either $\bX$ or $\bX^*$. However, it the technically correct first PC is $\tilde{\bx}^*$ rather than $\tilde{\bx}$.

This first PC is then removed from the data, and the procedure is repeated until all possible sample PCs are constructed. This is accomplished by calculating the product of $\bu_{m \times 1}$ and $\tilde{\pmb{x}}^*_{1 \times n}$, and subtracting it from $\bX^*$: 
$$
\bX^* - \bu \tilde{\pmb{x}}^* \ .
$$

## Calculating All PCs

All of the PCs can be calculated simultaneously. First, we construct the $m \times m$ sample covariance matrix $\bS$ with $(i,j)$ entry
$$
s_{ij} = \frac{\sum_{k=1}^n (x_{ik} - \bar{x}_{i\cdot})(x_{jk} - \bar{x}_{j\cdot})}{n-1}.
$$
The sample covariance can also be calculated by
$$
\bS = \frac{1}{n-1} \bX^{*} \bX^{*T}.
$$

It can be shown that 
$$
s^2_{\tilde{\bx}^*} = \bu^T \bS \bu,
$$
so identifying $\bu$ that maximizes $s^2_{\tilde{\bx}}$ also maximizes $\bu^T \bS \bu$.

Using a Lagrange multiplier, we wish to maximize

$$
\bu^T \bS \bu + \lambda(\bu^T \bu - 1).
$$


Differentiating with respect to $\bu$ and setting this to $\bO$, we get $\bS \bu - \lambda \bu = 0$ or

$$
\bSig \bu = \lambda \bu.
$$

For any such $\bu$ and $\lambda$ where this holds, note that

$$
s_{ij} = \frac{\sum_{k=1}^n (x_{ik} - \bar{x}_{i\cdot})(x_{jk} - \bar{x}_{j\cdot})}{n-1} = \bu^T \bSig \bu = \lambda
$$

so the PC's variance is $\lambda$.


The eigendecompositon of a matrix identifies all such solutions to $\bS \bu = \lambda \bu.$  Specifically, it calculates the decompositon

$$
\bS = \bU \bLambda \bU^T
$$

where $\bU$ is an $m \times m$ orthogonal matrix and $\bLambda$ is a diagonal matrix with entries $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_m \geq 0$.  

The fact that $\bU$ is orthogonal means $\bU \bU^T = \bU^T \bU = \bI$.  


The following therefore hold:

- For each column $j$ of $\bU$, say $\bu_j$, it follows that $\bS \bu_j = \lambda_j \bu_j$
- $\| \bu_j \|^2_2 = 1$ and $\bu_j^T \bu_k = \bO$ for $\lambda_j \not= \lambda_k$
- $\V(\bu_j^T \bX) = \lambda_j$
- $\V(\bu_1^T \bX) \geq \V(\bu_2^T \bX) \geq \cdots \geq \V(\bu_m^T \bX)$
- $\bS = \sum_{j=1}^m \lambda_j \bu_j \bu_j^T$
- For $\lambda_j \not= \lambda_k$, 
$$\Cov(\bu_j^T \bX, \bu_k^T \bX) = \bu_j^T \bS \bu_k = \lambda_k \bu_j^T \bu_k =  \bO$$ 

To calculate the actual principal components, let $x^*_{ij} = x_{ij} - \bar{x}_{i\cdot}$ be the mean-centered variables. Let $\bX^*$ be the matrix composed of these mean-centered variables.  Also, let $\bu_j$ be column $j$ of $\bU$ from $\bS = \bU \bLambda \bU^T$.

**Sample principal component** $j$ is then 

$$
\tilde{\bx}_j = \bu_j^T \bX^* = \sum_{i=1}^m u_{ij} \bx^*_i 
$$

for $j = 1, 2, \ldots, \min(m, n-1)$. For $j > \min(m, n-1)$, we have $\lambda_j = 0$, so these principal components are not necessary to calculate. The **loadings** corresponding to PC $j$ are $\bu_j$.

Note that the convention is that mean of PC $j$ is zero, i.e., that
$$
\frac{1}{n} \sum_{k=1}^n \tilde{x}_{jk} = 0,
$$
but as mentioned above $\sum_{i=1}^m u_{ij} \bx^*_i$ and the uncentered $\sum_{i=1}^m u_{ij} \bx_i$ have the same sample variance.

It can be calculated that the variance of PC $j$ is
$$
s^2_{\tilde{\bx}_j} = \frac{\sum_{k=1}^n \tilde{x}_{jk}^2}{n-1} = \lambda_j.
$$
The proportion of variance explained by PC $j$ is
$$
\operatorname{PVE}_j = \frac{\lambda_j}{\sum_{k=1}^m \lambda_k}.
$$

## Singular Value Decomposition

One way in which PCA is performed is to carry out a **singular value decomposition** (SVD) of the data matrix $\bX$.  Let $q = \min(m, n)$. Recalling that $\bX^*$ is the row-wise mean centered $\bX$, we can take the SVD of $\bX^*/\sqrt{n-1}$ to obtain

$$
\frac{1}{\sqrt{n-1}} \bX^* = \bU \bD \bV^T
$$

where $\bU_{m \times q}$, $\bV_{n \times q}$, and diagonal $\bD_{q \times q}$.  Also, we have the orthogonality properties $\bV^T \bV = \bU^T \bU = \bI_{q}$.  Finally, $\bD$ is composed of diagonal elements $d_1 \geq d_2 \geq \cdots \geq d_q \geq 0$ where $d_q = 0$ if $q = n$.


Note that

$$
\bS = \frac{1}{n-1} \bX^{*} \bX^{*T} = \bU \bD \bV^T \left(\bU \bD \bV^T\right)^T = \bU \bD^2 \bU^T.
$$

Therefore:

- The variance of PC $j$ is $\lambda_j = d_j^2$
- The loadings of PC $j$ are contained in the columns of the left-hand matrix from the decomposition of $\bS$ or $\bX^*$
- PC $j$ is row $j$ of $\bD \bV^T$

## A Simple PCA Function


```r
> pca <- function(x, space=c("rows", "columns"), 
+                 center=TRUE, scale=FALSE) {
+   space <- match.arg(space)
+   if(space=="columns") {x <- t(x)}
+   x <- t(scale(t(x), center=center, scale=scale))
+   x <- x/sqrt(nrow(x)-1)
+   s <- svd(x)
+   loading <- s$u
+   colnames(loading) <- paste0("Loading", 1:ncol(loading))
+   rownames(loading) <- rownames(x)
+   pc <- diag(s$d) %*% t(s$v)
+   rownames(pc) <- paste0("PC", 1:nrow(pc))
+   colnames(pc) <- colnames(x)
+   pve <- s$d^2 / sum(s$d^2)
+   if(space=="columns") {pc <- t(pc); loading <- t(loading)}
+   return(list(pc=pc, loading=loading, pve=pve))
+ }
```

The input is as follows:

- `x`: a matrix of numerical values
- `space`: either `"rows"` or `"columns"`, denoting which dimension contains the variables
- `center`: if `TRUE` then the variables are mean centered before calculating PCs
- `scale`: if `TRUE` then the variables are std dev scaled before calculating PCs


The output is a list with the following items:

- `pc`: a matrix of all possible PCs
- `loading`:  the weights or "loadings" that determined each PC
- `pve`: the proportion of variation explained by each PC

Note that the rows or columns of `pc` and `loading` have names to let you know on which dimension the values are organized.

## The Ubiquitous PCA Example

Here's an example very frequently encountered to explain PCA, but it's slightly complicated and conflates several ideas in PCA. I think it's not a great example to motivate PCA, but it's so common I want to carefully clarify what it's displaying.


```r
> set.seed(508)
> n <- 70
> z <- sqrt(0.8) * rnorm(n)
> x1 <- z + sqrt(0.2) * rnorm(n)
> x2 <- z + sqrt(0.2) * rnorm(n)
> X <- rbind(x1, x2)
> p <- pca(x=X, space="rows")
```


PCS is often explained by showing the following plot and stating, "The first PC finds the direction of maximal variance in the data..."

<img src="eda_files/figure-html/unnamed-chunk-178-1.png" width="672" style="display: block; margin: auto;" />

The above figure was made with the following code:


```r
> a1 <- p$loading[1,1] * p$pc[1,]
> a2 <- p$loading[2,2] * p$pc[2,]
> # a1 <- outer(p$loading[,1], p$pc[1,])[1,] + mean(x1)
> # a2 <- outer(p$loading[,1], p$pc[1,])[2,] + mean(x2)
> df <- data.frame(x1=c(x1, a1), 
+                  x2=c(x2, a2), 
+                  legend=c(rep("data",n),rep("pc1_projection",n)))
> ggplot(df) + geom_point(aes(x=x1,y=x2,color=legend)) + 
+   scale_color_manual(values=c("blue", "red"))
```

The red dots are therefore the projection of `x1` and `x2` onto the first PC, so they are neither the loadings nor the PC. This is rather complicated to understand before loadings and PCs are full understood.

Note that there are several ways to calculate these projections.

```
# all equivalent ways to get a1  
p$loading[1,1] * p$pc[1,]
outer(p$loading[,1], p$pc[1,])[1,] + mean(x1) 
lm(x1 ~ p$pc[1,])$fit # and

# all equivalent ways to get a2  
p$loading[2,2] * p$pc[2,]
outer(p$loading[,1], p$pc[1,])[2,] + mean(x2) 
lm(x2 ~ p$pc[1,])$fit
```

We haven't seen the `lm()` function yet, but once we do this example will be useful to revisit to understand what is meant  by "projection".


Here is PC1 vs PC2:


```r
> data.frame(pc1=p$pc[1,], pc2=p$pc[2,]) %>% 
+   ggplot() + geom_point(aes(x=pc1,y=pc2)) + 
+   theme(aspect.ratio=1)
```

<img src="eda_files/figure-html/unnamed-chunk-180-1.png" width="672" style="display: block; margin: auto;" />



Here is PC1 vs `x1`:


```r
> data.frame(pc1=p$pc[1,], x1=x1) %>% 
+   ggplot() + geom_point(aes(x=pc1,y=x1)) + 
+   theme(aspect.ratio=1)
```

<img src="eda_files/figure-html/unnamed-chunk-181-1.png" width="672" style="display: block; margin: auto;" />


Here is PC1 vs `x2`:


```r
> data.frame(pc1=p$pc[1,], x2=x2) %>% 
+   ggplot() + geom_point(aes(x=pc1,y=x2)) + 
+   theme(aspect.ratio=1)
```

<img src="eda_files/figure-html/unnamed-chunk-182-1.png" width="672" style="display: block; margin: auto;" />


Here is PC1 vs `z`:


```r
> data.frame(pc1=p$pc[1,], z=z) %>% 
+   ggplot() + geom_point(aes(x=pc1,y=z)) + 
+   theme(aspect.ratio=1)
```

<img src="eda_files/figure-html/unnamed-chunk-183-1.png" width="672" style="display: block; margin: auto;" />

## PC Biplots

Sometimes it is informative to plot a PC versus another PC.  This is called a **PC biplot**.

It is possible that interesting subgroups or clusters of *observations* will emerge.  

## PCA Examples

### Weather Data

These daily temperature data (in tenths of degrees C) come from meteorogical observations for weather stations in the US for the year 2012 provided by NOAA (National Oceanic and Atmospheric Administration).:


```r
> load("./data/weather_data.RData")
> dim(weather_data)
[1] 2811   50
> 
> weather_data[1:5, 1:7]
                  11       16  18       19  27  30       31
AG000060611 138.0000 175.0000 173 164.0000 218 160 163.0000
AGM00060369 158.0000 162.0000 154 159.0000 165 125 171.0000
AGM00060425 272.7619 272.7619 152 163.0000 163 108 158.0000
AGM00060444 128.0000 102.0000 100 111.0000 125  33 125.0000
AGM00060468 105.0000 122.0000  97 263.5714 155  52 263.5714
```

This matrix contains temperature data on 50 days and 2811 stations that were randomly selected.


First, we will convert temperatures to Fahrenheit:


```r
> weather_data <- 0.18*weather_data + 32
> weather_data[1:5, 1:6]
                  11       16    18       19    27    30
AG000060611 56.84000 63.50000 63.14 61.52000 71.24 60.80
AGM00060369 60.44000 61.16000 59.72 60.62000 61.70 54.50
AGM00060425 81.09714 81.09714 59.36 61.34000 61.34 51.44
AGM00060444 55.04000 50.36000 50.00 51.98000 54.50 37.94
AGM00060468 50.90000 53.96000 49.46 79.44286 59.90 41.36
> 
> apply(weather_data, 1, median) %>% 
+   quantile(probs=seq(0,1,0.1))
        0%        10%        20%        30%        40%        50% 
  8.886744  49.010000  54.500000  58.460000  62.150000  65.930000 
       60%        70%        80%        90%       100% 
 69.679318  73.490000  77.990000  82.940000 140.000000 
```

Let's perform PCA on these data.


```r
> mypca <- pca(weather_data, space="rows")
> 
> names(mypca)
[1] "pc"      "loading" "pve"    
> dim(mypca$pc)
[1] 50 50
> dim(mypca$loading)
[1] 2811   50
```


```r
> mypca$pc[1:3, 1:3]
            11        16         18
PC1 19.5166741 25.441401 25.9023874
PC2 -2.6025225 -4.310673  0.9707207
PC3 -0.6681223 -1.240748 -3.7276658
> mypca$loading[1:3, 1:3]
                Loading1    Loading2     Loading3
AG000060611 -0.015172744 0.013033849 -0.011273121
AGM00060369 -0.009439176 0.016884418 -0.004611284
AGM00060425 -0.015779138 0.007026312 -0.009907972
```


PC1 vs Time:  


```r
> day_of_the_year <- as.numeric(colnames(weather_data))
> data.frame(day=day_of_the_year, PC1=mypca$pc[1,]) %>%
+   ggplot() + geom_point(aes(x=day, y=PC1), size=2)
```

<img src="eda_files/figure-html/unnamed-chunk-188-1.png" width="672" style="display: block; margin: auto;" />



PC2 vs Time:  


```r
> data.frame(day=day_of_the_year, PC2=mypca$pc[2,]) %>%
+   ggplot() + geom_point(aes(x=day, y=PC2), size=2)
```

<img src="eda_files/figure-html/unnamed-chunk-189-1.png" width="672" style="display: block; margin: auto;" />



PC1 vs PC2 Biplot:  

This does not appear to be subgroups or clusters in the weather data set biplot of PC1 vs PC2.



```r
> data.frame(PC1=mypca$pc[1,], PC2=mypca$pc[2,]) %>%
+   ggplot() + geom_point(aes(x=PC1, y=PC2), size=2)
```

<img src="eda_files/figure-html/unnamed-chunk-190-1.png" width="672" style="display: block; margin: auto;" />


Proportion of Variance Explained:  


```r
> data.frame(Component=1:length(mypca$pve), PVE=mypca$pve) %>%
+   ggplot() + geom_point(aes(x=Component, y=PVE), size=2)
```

<img src="eda_files/figure-html/unnamed-chunk-191-1.png" width="672" style="display: block; margin: auto;" />


We can multiple the loadings matrix by the PCs matrix to reproduce the data:

```r
> # mean centered weather data
> weather_data_mc <- weather_data - rowMeans(weather_data)
> 
> # difference between the PC projections and the data
> # the small sum is just machine imprecision
> sum(abs(weather_data_mc/sqrt(nrow(weather_data_mc)-1) - 
+           mypca$loading %*% mypca$pc))
[1] 1.329755e-10
```


The sum of squared weights -- i.e., loadings -- equals one for each component:


```r
> sum(mypca$loading[,1]^2)
[1] 1
> 
> apply(mypca$loading, 2, function(x) {sum(x^2)})
 Loading1  Loading2  Loading3  Loading4  Loading5  Loading6  Loading7 
        1         1         1         1         1         1         1 
 Loading8  Loading9 Loading10 Loading11 Loading12 Loading13 Loading14 
        1         1         1         1         1         1         1 
Loading15 Loading16 Loading17 Loading18 Loading19 Loading20 Loading21 
        1         1         1         1         1         1         1 
Loading22 Loading23 Loading24 Loading25 Loading26 Loading27 Loading28 
        1         1         1         1         1         1         1 
Loading29 Loading30 Loading31 Loading32 Loading33 Loading34 Loading35 
        1         1         1         1         1         1         1 
Loading36 Loading37 Loading38 Loading39 Loading40 Loading41 Loading42 
        1         1         1         1         1         1         1 
Loading43 Loading44 Loading45 Loading46 Loading47 Loading48 Loading49 
        1         1         1         1         1         1         1 
Loading50 
        1 
```

PCs by contruction have sample correlation equal to zero:


```r
> cor(mypca$pc[1,], mypca$pc[2,])
[1] 3.135149e-17
> cor(mypca$pc[1,], mypca$pc[3,])
[1] 2.273613e-16
> cor(mypca$pc[1,], mypca$pc[12,])
[1] -1.231339e-16
> cor(mypca$pc[5,], mypca$pc[27,])
[1] -2.099516e-17
> # etc...
```

I can transform the top PC back to the original units to display it at a scale that has a more direct interpretation.


```r
> day_of_the_year <- as.numeric(colnames(weather_data))
> y <- -mypca$pc[1,] + mean(weather_data)
> data.frame(day=day_of_the_year, max_temp=y) %>%
+   ggplot() + geom_point(aes(x=day, y=max_temp))
```

<img src="eda_files/figure-html/unnamed-chunk-195-1.png" width="672" style="display: block; margin: auto;" />

### Yeast Gene Expression

Yeast cells were synchronized so that they were on the same approximate cell cycle timing in [Spellman et al. (1998)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC25624/).  The goal was to understand how gene expression varies over the cell cycle from a genome-wide perspective.


```r
> load("./data/spellman.RData")
> time
 [1]   0  30  60  90 120 150 180 210 240 270 330 360 390
> dim(gene_expression)
[1] 5981   13
> gene_expression[1:6,1:5]
                  0         30         60        90        120
YAL001C  0.69542786 -0.4143538  3.2350520 1.6323737 -2.1091820
YAL002W -0.01210662  3.0465649  1.1062193 4.0591467 -0.1166399
YAL003W -2.78570526 -1.0156981 -2.1387564 1.9299681  0.7797033
YAL004W  0.55165887  0.6590093  0.5857847 0.3890409 -1.0009777
YAL005C -0.53191556  0.1577985 -1.2401448 0.8170350 -1.3520947
YAL007C -0.86693416 -1.1642322 -0.6359588 1.1179131  1.9587021
```

Proportion Variance Explained:  


```r
> p <- pca(gene_expression, space="rows")
> ggplot(data.frame(pc=1:13, pve=p$pve)) + 
+   geom_point(aes(x=pc,y=pve), size=2)
```

<img src="eda_files/figure-html/unnamed-chunk-197-1.png" width="672" style="display: block; margin: auto;" />


PCs vs Time (with Smoothers):  

<img src="eda_files/figure-html/unnamed-chunk-198-1.png" width="672" style="display: block; margin: auto;" />

### HapMap Genotypes

I curated a small data set that cleanly separates human subpopulations from the [HapMap](https://en.wikipedia.org/wiki/International_HapMap_Project) data.  These include unrelated individuals from Yoruba people from Ibadan, Nigeria (YRI), Utah residents of northern and western European ancestry (CEU), Japanese individuals from Tokyo, Japan (JPT), and Han Chinese individuals from Beijing, China (CHB). 


```r
> hapmap <- read.table("./data/hapmap_sample.txt")
> dim(hapmap)
[1] 400  24
> hapmap[1:6,1:6]
           NA18516 NA19138 NA19137 NA19223 NA19200 NA19131
rs2051075        0       1       2       1       1       1
rs765546         2       2       0       0       0       0
rs10019399       2       2       2       1       1       2
rs7055827        2       2       1       2       0       2
rs6943479        0       0       2       0       1       0
rs2095381        1       2       1       2       1       1
```

Proportion Variance Explained:  


```r
> p <- pca(hapmap, space="rows")
> ggplot(data.frame(pc=(1:ncol(hapmap)), pve=p$pve)) + 
+   geom_point(aes(x=pc,y=pve), size=2)
```

<img src="eda_files/figure-html/unnamed-chunk-200-1.png" width="672" style="display: block; margin: auto;" />

PC1 vs PC2 Biplot:  

<img src="eda_files/figure-html/unnamed-chunk-201-1.png" width="672" style="display: block; margin: auto;" />


PC1 vs PC3 Biplot:  

<img src="eda_files/figure-html/unnamed-chunk-202-1.png" width="672" style="display: block; margin: auto;" />


PC2 vs PC3 Biplot:  

<img src="eda_files/figure-html/unnamed-chunk-203-1.png" width="672" style="display: block; margin: auto;" />

