## Main File for Pricing Game
## Loading the training_set and the testing_set
download.file("https://raw.githubusercontent.com/fpechon/SummerSchool/master/PricingGame/training_set.RData", "training_set.RData")
load("training_set.RData")


### Your model comes here (for instance a GLM, with DriverAge < 50 and a gamma distribution)

training_set$DriverAgeCAT = (training_set$DriverAge < 50)
model = glm(AvClaimAmount ~ DriverAgeCAT,
            data=training_set,
            family = Gamma(link="log"))



### The testing set. Does not have the average claim cost (that you have to predict).
download.file("https://raw.githubusercontent.com/fpechon/SummerSchool/master/PricingGame/testing_set.RData", "testing_set.RData")
load("testing_set.RData")

#If we added variables, we should do it on the testing_set as well !
testing_set$DriverAgeCAT = (testing_set$DriverAge < 50)


# predict the average cost of claims on the testing_set (may have to be adapted depending on your model,
# here we need to take the exponential)
output = data.frame(PolicyID = testing_set$PolicyID)
output$pred = predict(model, testing_set, type="response")
all(output$pred>=0) # Check if your predictions are on the correct scale (for a GLM/GAM for instance)
head(output) # Look at 6 first predictions.


# Save and send me the predictions. There should be 2760 predictions.
if(length(output$pred) == nrow(testing_set)){
  print("Correct number of predictions")
}else{
  print("Incorrect number of predictions")
}

#pred is the file to send me

save(output, file="PG_YOURNAME.RData") #Send me this file.

sqrt(sum((testing_set$AvClaimAmount - output$pred)^2)) ## Will be 0 since the AvClaimCost not given in testing_set.
