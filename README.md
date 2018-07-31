README
================

SummerSchool
============

All the scripts of the UseR sessions will be available here.

More details on the Pricing Game will follow.

Please run the following script in your R session to install all the required packages. Make sure Rtools (Windows) or gfortan (Mac) has been installed previously.

``` r
if (!require("xts")) install.packages("xts")
if (!require("zoo")) install.packages("zoo")
if (!require("sp")) install.packages("sp")
if (!require("CASdatasets")) install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/R/")
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


require("CASdatasets")
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

install_github("gbm-developers/gbm3")
install_github("fpechon/rfCountData")
require("gbm3")
require("rfCountData")
```
