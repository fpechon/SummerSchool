README
================

SummerSchool
============

All the scripts of the UseR sessions will be available here.

More details on the Pricing Game will follow.

Required Software
-----------------

-   R (at least 3.4.3 <https://cloud.r-project.org/bin/windows/base/old/3.4.3> )
-   RStudio ( <https://www.rstudio.com/products/rstudio/download/#download> )
-   For Windows, Rtools : <https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe>
-   For Mac: Install gfortran-6.1.pkg (more info <https://cran.r-project.org/bin/macosx/tools/> )

Please run the following script in your R session to install all the required packages. Make sure Rtools (Windows) or gfortan (Mac) has been installed previously.

Start with the package containing datasets. If there is a query to install from source, answer "y" (yes).

``` r
if (!require("CASdatasets")) install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/R/")
require("CASdatasets")
```

If the installation succeeded, the following lines should load a dataset and show the first six lines.

``` r
data("freMTPLfreq")
head(freMTPLfreq)
```

    ##   PolicyID ClaimNb Exposure Power CarAge DriverAge
    ## 1        1       0     0.09     g      0        46
    ## 2        2       0     0.84     g      0        46
    ## 3        3       0     0.52     f      2        38
    ## 4        4       0     0.45     f      2        38
    ## 5        5       0     0.15     g      0        41
    ## 6        6       0     0.75     g      0        41
    ##                                Brand     Gas Region Density
    ## 1 Japanese (except Nissan) or Korean  Diesel    R72      76
    ## 2 Japanese (except Nissan) or Korean  Diesel    R72      76
    ## 3 Japanese (except Nissan) or Korean Regular    R31    3003
    ## 4 Japanese (except Nissan) or Korean Regular    R31    3003
    ## 5 Japanese (except Nissan) or Korean  Diesel    R52      60
    ## 6 Japanese (except Nissan) or Korean  Diesel    R52      60

Then, we can install and load all the other packages.

``` r
if (!require("xts")) install.packages("xts")
if (!require("Rcpp")) install.packages("Rcpp")
if (!require("zoo")) install.packages("zoo")
if (!require("sp")) install.packages("sp")
if (!require("caret")) install.packages("caret")
if (!require("mgcv")) install.packages("mgcv")
if (!require("plyr")) install.packages("plyr")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("visreg")) install.packages("visreg")
if (!require("MASS")) install.packages("MASS")
if (!require("plotrix")) install.packages("plotrix")
if (!require("lme4")) install.packages("lme4")
if (!require("glmnet")) install.packages("glmnet")
if (!require("parallel")) install.packages("parallel")
if (!require("devtools")) install.packages("devtools")
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("randomForest")) install.packages("randomForest")
if (!require("xgboost")) install.packages("xgboost")

require("Rcpp")
require("mgcv")
require("caret")
require("gridExtra")
require("plyr")
require("visreg")
require("MASS")
require("plotrix")
require("lme4")
require("glmnet")
require("parallel")
require("devtools")
require("rpart")
require("rpart.plot")
require("xgboost")

install_github("gbm-developers/gbm3")
install_github("fpechon/rfCountData")
require("gbm3")
require("rfCountData")
```
