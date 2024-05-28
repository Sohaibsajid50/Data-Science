# Load the dataset
library(datasets)
data("iris")
iris <- datasets::iris
View(iris)


# Display Summary Statistics
head(iris,4)
tail(iris)

#summary
summary(iris)
summary(iris$Sepal.Length)

# check to see if there is any missing data
sum(is.na(iris))

# skimr() - expands on summary() by providing larger set of statistics
install.packages("skimr")
# https://github.com/ropensci/skimr

library(skimr)

skim(iris) # Perform skim to display summary statistics

# Group data by Species then perform skim
iris %>% 
  dplyr::group_by(Species) %>% 
  skim() 


# Data Visualization 

# Panel PLot
plot(iris)
plot(iris, col = 'red')

#scatter plot
plot(iris$Sepal.Width, iris$Sepal.Length)      

plot(iris$Sepal.Width, iris$Sepal.Length, col = "blue")    # make blue circles

plot(iris$Sepal.Width, iris$Sepal.Length, col = "red",
     xlab = " Sepal Width", ylab = "Sepal Length")    # make red circles, add x and y axis label

# Histogram
hist(iris$Sepal.Width)
hist(iris$Sepal.Width, col = "red")     

# Feature plot
# https://www.machinelearningplus.com/machine-learning/caret-package/

install.packages("caret")
library(caret)
library(lattice)
featurePlot(x = iris[,1:4], y = iris$Species,
            plot = "box",
            strip = strip.custom(par.strip.text = list(cex=0.7)),
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free")))