---
description: 
title: TS - missing data imputation
categories: ['Time Series']
tags: ['missing data','time series','imputation']
date: '2022-10-26'
toc: true
---










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
sort(rand.unemp[ , lapply(.SD, function(x) mean((x - unemp$UNRATE)^2, na.rm = TRUE)),
             .SDcols = c("impute.ff", "impute.rm.nolookahead", "impute.rm.lookahead", "impute.li")])
##      impute.ff impute.rm.nolookahead  impute.li impute.rm.lookahead
## 1: 0.006681514           0.008851728 0.02645122           0.1084783

sort(bias.unemp[ , lapply(.SD, function(x) mean((x - unemp$UNRATE)^2, na.rm = TRUE)),
             .SDcols = c("impute.ff", "impute.rm.nolookahead", "impute.rm.lookahead", "impute.li")])
##      impute.li impute.rm.lookahead   impute.ff impute.rm.nolookahead
## 1: 0.001443114          0.00358427 0.004109131           0.004140449


smoothed = unemp[, HoltWinters(UNRATE, alpha = 0.1, beta = FALSE, gamma = FALSE)]
```


```python
import pandas as pd

unemp = r.unemp


## We can use the pandas.DataFrame.ewm() function to calculate the exponentially weighted moving average for a certain number of previous periods.
```

alpha -  smoothing factor

`\(0 < \alpha \leq 1\)`


When adjust=False, the exponentially weighted function is calculated recursively:

`$$\begin{split}\begin{split}
y_0 &= x_0\\
y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
\end{split}\end{split}$$`


```python
unemp['Smooth.5'] = unemp.UNRATE.ewm(alpha=0.5,adjust=False).mean()
unemp['Smooth.1'] = unemp.UNRATE.ewm(alpha=0.1,adjust=False,).mean()
unemp['Smooth.9'] = unemp.UNRATE.ewm(alpha=0.9,adjust=False,).mean()

import matplotlib.pyplot as plt

plt.plot(unemp['unemp.UNRATE'], label='Sales')
## Error in py_call_impl(callable, dots$args, dots$keywords): KeyError: 'unemp.UNRATE'
plt.plot(unemp['Smooth.1'], label='4-day EWM')
plt.plot(unemp['Smooth.5'], label='Sales')
plt.plot(unemp['Smooth.9'], label='Sales')

#add legend to plot
plt.legend(loc=2)
plt.show()
```

<img src="/post/TS_missing_data_imputation_files/figure-html/unnamed-chunk-5-1.png" width="100%" />
