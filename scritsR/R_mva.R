#.libPaths(.Library)
#install.packages("vroom")  
#install.packages("readr")  
#install.packages("GGally")
#install.packages("ggplot2")
#install.packages("ggthemes")  


library(readr)
library(radiant)
library(ggplot2)
library(ggthemes)
library(GGally)
library(Hotelling)
library(reshape2)

stats <- read_csv("stats.csv", col_types = cols(...1 = col_skip(),
                                                image_id = col_skip(), cell_id = col_skip(),
                                                bethesda = col_integer(), FDN = col_skip(),
                                                FDC = col_skip()))

stats$bethesda <- as.factor(stats$bethesda)

### EDA - Exploratory Data Analysis
save(file="stats.Rdata", stats)
#radiant()
cols_to_plot <- c('bethesda', 'areaN', 'perimeterN','major_axisN', 'minor_axisN','equivalent_diameterN', 
                  'radial_distance_maxN', 'radial_distance_meanN', 'radial_distance_sdN', 'eccentricityN',
                  'circularityN', 'convexityN', 'solidityN', 'extentN', 'RAN', 'RIN', 'radial_distance_EN',
                  'radial_distance_kurtoseN','Use_curv1N', 'Use_curv2N', 'Use_curv3N', 'major_axis_angleN')
stats.molten <- reshape2::melt(stats[,cols_to_plot], id.vars = "bethesda")
ggplot(stats.molten, aes(x=variable, y=value, col=variable)) +
  geom_boxplot() +
  facet_wrap(bethesda ~.) +
  theme_tufte()

ggpairs(data=stats, columns=c(2,5:9), aes(colour = bethesda, alpha=.2))
#ggpairs(data=stats, columns=c(44:50), aes(colour = bethesda, alpha=.2))

### Classical Multivariate Analysis
#Class0 <- subset(stats[, c(2,5:9)], stats$bethesda=="0")
#Class1 <- subset(stats[, c(2,5:9)], stats$bethesda=="1")
#hotelling.test(x=Class0, y=Class1)


