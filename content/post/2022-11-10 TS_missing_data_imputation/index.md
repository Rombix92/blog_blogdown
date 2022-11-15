---
title: "TS - missing data imputation & Smoothing"
description: null
date: "2022-11-15"
tags: ['missing data','time series','imputation', 'python', 'R']
categories: ['Time Series', 'Smoothing']
toc: yes
---







## Dealing with missing data in Time Series


```r
require(zoo)
require(data.table)
library(dplyr)
library(lubridate)

unemp <- fread(paste0(data_file_path,"bezrobocie_USA.csv")) %>% data.table::melt( id.vars='Year',
                                                           variable.name = "months",
                                                           value.name='UNRATE') %>% left_join(
  data.frame(month_nr=c(1:12),
             months= c("Jan","Feb","Mar",
                       "Apr","May","Jun",
                       "Jul","Aug","Sep",
                       "Oct","Nov","Dec"))
) %>% mutate(DATE=as_date('0000-01-01',format = '%Y-%m-%d')+years(as.numeric(Year)) + months(month_nr-1)) 

head(unemp)
##    Year months UNRATE month_nr       DATE
## 1: 1948    Jan    3.4        1 1948-01-01
## 2: 1949    Jan    4.3        1 1949-01-01
## 3: 1950    Jan    6.5        1 1950-01-01
## 4: 1951    Jan    3.7        1 1951-01-01
## 5: 1952    Jan    3.2        1 1952-01-01
## 6: 1953    Jan    2.9        1 1953-01-01


unemp = unemp[, DATE := as.Date(DATE)][!is.na(UNRATE),.(DATE, UNRATE)]
setkey(unemp, DATE)

## Creating dataset with random missing values
rand.unemp.idx <- sample(1:nrow(unemp), .1*nrow(unemp))
rand.unemp <- unemp[-rand.unemp.idx]

## Creating dataset with systematical missing values, appearing in month with highest unemployment rate
high.unemp.idx <- which(unemp$UNRATE > 8)
high.unemp.idx <- sample(high.unemp.idx, .5 * length(high.unemp.idx))
bias.unemp <- unemp[-high.unemp.idx]


## to identyfy missing data I wil use rolling joins tool from data.table package    
all.dates <- seq(from = unemp$DATE[1], to = tail(unemp$DATE, 1), by = "months")
rand.unemp = rand.unemp[J(all.dates), roll=FALSE]
bias.unemp = bias.unemp[J(all.dates), roll=FALSE]

## forward filling
rand.unemp[, impute.ff := na.locf(UNRATE, na.rm = FALSE)]
bias.unemp[, impute.ff := na.locf(UNRATE, na.rm = FALSE)]

## Mean moving average with use of lookahead phenomen
rand.unemp[, impute.rm.lookahead := rollapply(data=c(UNRATE,NA, NA), width=3,
          FUN= function(x) {
                         if (!is.na(x[1])) x[1] else mean(x, na.rm = TRUE)
                         })]         
bias.unemp[, impute.rm.lookahead := rollapply(c(UNRATE, NA,NA), 3,
            FUN= function(x) {
                         if (!is.na(x[1])) x[1] else mean(x, na.rm = TRUE)
                         })]         



## Mean moving average withou use of lookahead phenomen
rand.unemp[, impute.rm.nolookahead := rollapply(c(NA, NA, UNRATE), 3,
             function(x) {
                         if (!is.na(x[3])) x[3] else mean(x, na.rm = TRUE)
                         })]         
bias.unemp[, impute.rm.nolookahead := rollapply(c(NA, NA, UNRATE), 3,
             function(x) {
                         if (!is.na(x[3])) x[3] else mean(x, na.rm = TRUE)
                         })]    





## linear interpolation fullfilling NA with linear interpolation between two data points
rand.unemp[, impute.li := na.approx(UNRATE, maxgap=Inf)]
bias.unemp[, impute.li := na.approx(UNRATE)]

zz <- c(NA, 9, 3, NA, 3, 2,NA,5,6,10,NA,NA,NA,0)
na.approx(zz, na.rm = FALSE, maxgap=2)
##  [1]   NA  9.0  3.0  3.0  3.0  2.0  3.5  5.0  6.0 10.0   NA   NA   NA  0.0
na.approx(zz, na.rm = FALSE, maxgap=Inf)
##  [1]   NA  9.0  3.0  3.0  3.0  2.0  3.5  5.0  6.0 10.0  7.5  5.0  2.5  0.0
na.approx(zz,xout=11, na.rm = FALSE, maxgap=Inf)
## [1] 7.5





## Using root mean square error to compare methods
print(rand.unemp[ , lapply(.SD, function(x) mean((x - unemp$UNRATE)^2, na.rm = TRUE)),
             .SDcols = c("impute.ff", "impute.rm.nolookahead", "impute.rm.lookahead", "impute.li")])
##      impute.ff impute.rm.nolookahead impute.rm.lookahead   impute.li
## 1: 0.003363029           0.004768931         0.004799555 0.001306917

print(bias.unemp[ , lapply(.SD, function(x) mean((x - unemp$UNRATE)^2, na.rm = TRUE)),
             .SDcols = c("impute.ff", "impute.rm.nolookahead", "impute.rm.lookahead", "impute.li")])
##    impute.ff impute.rm.nolookahead impute.rm.lookahead impute.li
## 1: 0.2957906             0.2217154          0.01558493  0.186025
```

## Smoothing

Smoothing is commonelly used forecasting method. Smoothed time series can be used as zero hypothesis to for testing more sophisticated methods.


```python
import pandas as pd
import numpy as np
import datetime

unemp = r.unemp
#unemp.index = unemp.DATE

df = unemp.copy()
df = df[((df.DATE >=pd.to_datetime('2014-01-01')) & (df.DATE < pd.to_datetime('2019-01-01')))]
## /Users/lrabalski1/miniforge3/envs/everyday_use/lib/python3.8/site-packages/pandas/core/ops/array_ops.py:73: FutureWarning: Comparison of Timestamp with datetime.date is deprecated in order to match the standard library behavior.  In a future version these will be considered non-comparable.Use 'ts == pd.Timestamp(date)' or 'ts.date() == date' instead.
##   result = libops.scalar_compare(x.ravel(), y, op)
df = df.rename(columns={"UNRATE": "data"})
#df.reset_index(drop=True, inplace=True)


train = df[['data']].iloc[:-12, :]
test = df[['data']].iloc[-12:, :]
# train.index = pd.to_datetime(train.index)
# test.index = pd.to_datetime(test.index)
## We can use the pandas.DataFrame.ewm() function to calculate the exponentially weighted moving average for a certain number of previous periods.
```

### moving average

An improvement over simple average is the average of n last points. Obviously the thinking here is that only the recent values matter. Calculation of the moving average involves what is sometimes called a "sliding window" of size n:


```python

def average(series):
    return float(sum(series))/len(series)

# moving average using n last points
def moving_average(series, n):
    return average(series[-n:])

moving_average(train.data,4)
## 4.2
```

### Weighted Moving Average

A weighted moving average is a moving average where within the sliding window values are given different weights, typically so that more recent points matter more.

Instead of selecting a window size, it requires a list of weights ([**which should add up to 1**]{.underline}). For example if we picked [0.1, 0.2, 0.3, 0.4] as weights, we would be giving 10%, 20%, 30% and 40% to the last 4 points respectively. In Python:


```python
# weighted average, weights is a list of weights
def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result
  
weights = [0.1, 0.15, 0.25, 0.5]
weighted_average(train.data.values, weights)

## 4.16
```

### exponentially weightening

The exponentially weighted function is calculated recursively:

`$$\begin{split}\begin{split}
y_0 &= x_0\\
y_t &= \alpha x_t + (1 - \alpha) y_{t-1} ,
\end{split}\end{split}$$`

where alpha is smoothing factor `\(0 < \alpha \leq 1\)` . The higher the α, the faster the method "forgets".

There is an aspect of this method that programmers would appreciate that is of no concern to mathematicians: it's simple and efficient to implement. Here is some Python. Unlike the previous examples, this function returns expected values for the whole series, not just one point.


```python
# given a series and alpha, return series of smoothed points
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


res_exp_smooth8 = exponential_smoothing(train.data.values, alpha=0.8)
res_exp_smooth5 = exponential_smoothing(train.data.values, alpha=0.5)
res_exp_smooth2 = exponential_smoothing(train.data.values, alpha=0.2)

```





### Conclusion

I showed some basic forecasting methods: moving average, weighted moving average and, finally, single exponential smoothing. One very important characteristic of all of the above methods is that remarkably, they can only forecast a single point. That's correct, just one.

### Double exponential smoothing

a.k.a Holt Method

In case of forecasting simple exponential weightening isn't giving good results for data posessing longterm trend. For this purpose it is good to apply method aimed for data with trend (Holt) or with trend and seasonality (Holt-Winter).

Double exponential smoothing is nothing more than exponential smoothing applied to both level and trend.


```python

# given a series and alpha, return series of smoothed points
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level= level
        level =  alpha*value + (1-alpha)*(last_level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


res_double_exp_smooth_alpha_9_beta9=double_exponential_smoothing(train.data.values, alpha=0.9, beta=0.9)
```

### Triple Exponential Smoothing

a.k.a Holt-Winters Method

#### Initial Trend

For double exponential smoothing we simply used the first two points for the initial trend. With seasonal data we can do better than that, since we can observe many seasons and can extrapolate a better starting trend. The most common practice is to compute the average of trend averages across seasons.


```python

def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

res_initial = initial_trend(train.data.values,12)
res_initial
## -0.07361111111111113
```

The value of -0.074 can be interpreted that unemployment rate between first two years change on average by -0.074  between each pair of the same month.

#### Initial Seasonal Components

The situation is even more complicated when it comes to initial values for the seasonal components. Briefly, we need to

1.  compute the average level for every observed season (in our case YEAR) we have,

2.  divide every observed value by the average for the season it's in

3.  and finally average each of these numbers across our observed seasons.


```python
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

initial_seasonal_components(train.data.values,12)
## {0: 0.2833333333333332, 1: 0.2583333333333331, 2: 0.20833333333333304, 3: 0.10833333333333317, 4: 0.10833333333333339, 5: -0.01666666666666683, 6: -0.04166666666666696, 7: -0.04166666666666674, 8: -0.11666666666666714, 9: -0.216666666666667, 10: -0.21666666666666679, 11: -0.3166666666666669}
```

Seasonal values we can interpret as average distance value from seasonal average. We can see that January {0} is on higher than average and December value {11} is lower than average. We can see that those month differ from each other exactly with the power of those values


```python
df[pd.to_datetime(df.DATE).dt.month.isin([1,12])]
##            DATE  data
## 792  2014-01-01   6.6
## 803  2014-12-01   5.6
## 804  2015-01-01   5.7
## 815  2015-12-01   5.0
## 816  2016-01-01   4.8
## 827  2016-12-01   4.7
## 828  2017-01-01   4.7
## 839  2017-12-01   4.1
## 840  2018-01-01   4.0
## 851  2018-12-01   3.9
```




```python
def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

res_triple_exp_smooth = triple_exponential_smoothing(train.data.values, 12, 0.7, 0.02, 0.9, 10)
```

A Note on α, β and γ

You may be wondering from where values 0.7, 0.02 and 0.9 for α, β and γ It was done by way of trial and error: simply running the algorithm over and over again and selecting the values that give you the smallest SSE. This process is known as fitting.

There are more efficient methods at zooming in on best values. One good algorithm for this is Nelder-Mead, which is what tgres uses.


### fitting data


```python
res = [res_exp_smooth8,res_exp_smooth5,res_exp_smooth2,res_double_exp_smooth_alpha_9_beta9,res_triple_exp_smooth]
RMSE = []
i=1
for i in range(len(res)):
  RMSE.append(np.sqrt(np.mean(np.square((train.data.values[0:48]- res[i][0:48])))))
RMSE
## [0.029444314294186542, 0.08240165108170797, 0.2272843505233408, 0.10916100840899878, 0.06909009711283208]
```

In case of fitting smoothed data to raw data, the best fit possess single exponenetial smoothing method with **alpha =0.8** (putting higher weight on most recent data). This is exactly what could be expected. Is it then the best **forecasting method** for my data?

Obviously not.

Since all method take data point from time *t* for estimating smoothed value for time *t* such a models are not forecasting one's. We are dealing here with **lookahead** problem. In order to predict we are using data which shouldn't be available at the moment of making prediction.

Out of three methods prediction capabilities posses Holt method (using trend to linearly predict further data points) and Holt-Winter method (using trend and seasonality to predict further data points).

### plot


```python
import matplotlib.pyplot as plt
import datetime
plt.style.use('Solarize_Light2')

   
plt.clf()
fig = plt.figure(figsize=(5,8))
# f.set_figwidth(10)#inches
# f.set_figheight(20)#inches
ax1 = fig.add_subplot(5, 1, 1) 
plt.plot(train.data.values, label='raw')
plt.plot(res_exp_smooth8, label='exp_smooth_alpha_0.8')
ax2 =fig.add_subplot(5, 1, 2)
plt.plot(train.data.values, label='raw')
plt.plot(res_exp_smooth5, label='exp_smooth_alpha_0.5')
ax3 =fig.add_subplot(5, 1, 3)
plt.plot(train.data.values, label='raw')
plt.plot(res_exp_smooth2, label='exp_smooth_alpha_0.2')
ax4 =fig.add_subplot(5, 1, 4)
plt.plot(train.data.values, label='raw')
plt.plot(res_double_exp_smooth_alpha_9_beta9, label='res_double_exp_smooth_alpha_9_beta9')
ax5 =fig.add_subplot(5, 1, 5)
plt.plot(train.data.values, label='raw')
plt.plot(res_triple_exp_smooth, label='res_triple_exp_smooth')
ax1.set_title('raw data vs exponential forecast')
ax1.legend(loc="upper left")
ax2.legend(loc="upper left")
ax3.legend(loc="upper left")
ax4.legend(loc="upper left")
ax5.legend(loc="upper left")
ax1.sharex(ax5)
ax2.sharex(ax5)
ax3.sharex(ax5)
ax4.sharex(ax5)

fig.tight_layout()
fig.savefig('index_files/figure-html/unnamed-chunk-15-1.png', bbox_inches='tight')

plt.show()

```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-16-1.png" width="100%" />
