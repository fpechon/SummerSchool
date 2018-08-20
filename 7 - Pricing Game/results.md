Results Pricing Game
================

Please find below the results of the Pricing Game. The blue line corresponds to the null model (average claim cost of the training set as predictor on the testing set.).

``` r
library(ggplot2)
ggplot(leaderboard) + geom_point(aes(x=Letter, y=RMSE, color=Letter)) + xlab("Id Participant") + geom_line(aes(x=Letter, y=NullModel), color="blue")+scale_x_continuous(breaks=(1:20))+
  scale_color_gradient(low="green", high="red")+theme(legend.position = "none")+ggtitle("Score on testing set (RMSE Loss function)")
```

<img src="results_files/figure-markdown_github/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

The best model reached an RMSE as low as 565.21408 on the testing set.
