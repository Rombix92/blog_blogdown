---
description: R data.table
title: "R lib: data.table"
categories: ["data.table"]
tags: ["R", "data.table", "datagrapling","rolling joins"]
date: '2022-11-07'
toc: true
---



## Rolling join

[source](https://www.gormanalysis.com/blog/r-data-table-rolling-joins/)

### Problem

suppose you have a table of product sales and a table of commercials. You might want to associate each product sale with the most recent commercial that aired prior to the sale. In this case, you cannot do a basic join between the sales table and the commercials table because each sale was not tracked with a CommercialId attribute. Instead, you need to generate a mapping between sales and commercials based on logic involving their dates.


```r
library(dplyr)
library(data.table)
```


```r
sales <- data.table(
  SaleId = c("S1", "S2", "S3", "S4", "S5"),
  SaleDate = as.Date(c("2014-2-20", "2014-5-1", "2014-6-15", "2014-7-1", "2014-12-31"))
)

commercials <- data.table(
  CommercialId = c("C1", "C2", "C3", "C4"),
  CommercialDate = as.Date(c("2014-1-1", "2014-4-1", "2014-7-1", "2014-9-15"))
)

sales[, RollDate := SaleDate]
commercials[, RollDate := CommercialDate]

setkey(sales, "RollDate")
setkey(commercials, "RollDate")
```

data.table is associating each commercial with the most recent sale prior to the commercial date (and including the commercial date). In other words, the most recent sale prior to each commercial is said to "roll forward", and the SaleDate is mapped to the CommercialDate. Notice that sale S4 was the most recent sale prior to commercial C3 and C4, so S4 appears twice in the resultant table.

THIS HAVE NO logical SENSE. go forward


```r
sales[commercials, roll = TRUE]
```

```
##    SaleId   SaleDate   RollDate CommercialId CommercialDate
## 1:   <NA>       <NA> 2014-01-01           C1     2014-01-01
## 2:     S1 2014-02-20 2014-04-01           C2     2014-04-01
## 3:     S4 2014-07-01 2014-07-01           C3     2014-07-01
## 4:     S4 2014-07-01 2014-09-15           C4     2014-09-15
```

Here data.table is associating each sale with the most recent commercial prior to (or including) the SaleDate. This is the solution to our originally stated problem.


```r
commercials[sales, roll = TRUE]
```

```
##    CommercialId CommercialDate   RollDate SaleId   SaleDate
## 1:           C1     2014-01-01 2014-02-20     S1 2014-02-20
## 2:           C2     2014-04-01 2014-05-01     S2 2014-05-01
## 3:           C2     2014-04-01 2014-06-15     S3 2014-06-15
## 4:           C3     2014-07-01 2014-07-01     S4 2014-07-01
## 5:           C4     2014-09-15 2014-12-31     S5 2014-12-31
```

![](https://www.gormanalysis.com/blog/r-data-table-rolling-joins_files/roll-2.gif)

data.table also supports backward rolling joins by setting roll = -Inf.

```r
commercials[sales, roll = -Inf]
```

```
##    CommercialId CommercialDate   RollDate SaleId   SaleDate
## 1:           C2     2014-04-01 2014-02-20     S1 2014-02-20
## 2:           C3     2014-07-01 2014-05-01     S2 2014-05-01
## 3:           C3     2014-07-01 2014-06-15     S3 2014-06-15
## 4:           C3     2014-07-01 2014-07-01     S4 2014-07-01
## 5:         <NA>           <NA> 2014-12-31     S5 2014-12-31
```
