#Loading the required Packages:
library(tidyverse)
library(ggplot2) #use ggplot2 to add layer for visualization
library(skimr) #SKIM function - For statistical summary
library(GGally)
library(ggplot2)
library(corrplot)
library(caTools)
library(caret)
library(usmap) #import the package
library(stringr) #Rename variable names
library(scatterplot3d)  #Visualization
library(moments) #skewness
library(DataExplorer)
library(ggpubr)
library(EnvStats) #Roseners test for outliers
library(gclus) #Scatterplot Matrices from the glus Package 
library(plotly) #world map
library(readxl) #Read UN dataset
library(superml)
library(RColorBrewer) #Bar chart
library(base64)
library(ggpubr)
library(correlationfunnel)
library(randomForest)
library(e1071) #SVM
library(class)
library(datasets)
suppressWarnings
suppressMessages
options(error = expression(NULL))
#For Text Mining
library(tm)
library(wordcloud)
library(SnowballC)
library(quanteda)
library(EnvStats) #Roseners test for outliers
library(stringr) #to count len of a string

# Text Mining

## The dataset GoodReads.csv contains book descriptions. Load this dataset and draw random sample of size 5000 from it, setting the seed to 31.
#Text Extraction
data_main <-read.csv("GoodReads.csv", header=T, stringsAsFactors = TRUE)
#Drawing random sample of size 5000
set.seed(31)
data_goodread <- data_main[sample(nrow(data_main), 5000), ]

## Plot and describe the distribution of average ratings of the books.
#Mean of rating:
mean (data_goodread$rating)
#Plotting the rating distribution
ggplot(data_goodread,aes(x=rating)) + 
  geom_text(aes(x=4.1, y=.8, label='Rating'), size=3)+
  geom_vline(aes(xintercept=mean(rating)), linetype='dashed')+
  geom_density(alpha= 0.3,fill='blue')+ 
  ggtitle('Rating Destribution plot')

#Insight: Average value of feature rating is 3.828692. From the distribution graph we can observe that the data points are left skewed, which means most of the value lie between 3 and 5.

## Clean the data in your dataframe. Explain your reasoninng for selecting the cleaning methods you decided to use.

#Analysing the structure of the dataset:
data_goodread %>% glimpse()

#Selecting only relevant Values
data_goodread <- data_goodread %>% select(c(desc,rating))

#Insight : Only relevant features desc and rating is selected for further text analysis.

#Check NA values
apply(is.na(data_goodread),2, sum)

#Insight : From the above table we can see that there are no Null values.

## Categorize the rating variable into several categories. Explain how many categories you decided to have, where you drew the line, and why you decided to do it this way

data_goodread <- mutate (data_goodread
                         , rating_category = case_when(
                           data_goodread$rating > 0 & data_goodread$rating <= 1 ~ 1
                           , data_goodread$rating > 1 & data_goodread$rating <= 2 ~ 2
                           , data_goodread$rating > 2 & data_goodread$rating <= 3 ~ 3
                           , data_goodread$rating > 3 & data_goodread$rating <= 4 ~ 4
                           , data_goodread$rating > 4 ~ 5))
rating_grouped <- prop.table(table(data_goodread$rating_category))
rating_grouped <- as.data.frame(rating_grouped)
rating_grouped

#Insight: Considering the rating value which lies from 0 to 5. I have categorized the observations in to 5 category as 1,2,3,4,5.
#Each category is assigned under standard specific range.

hsize <- 4
rating_grouped_pie <- rating_grouped %>% mutate(x = hsize)
ggplot(rating_grouped_pie, aes(x = hsize, y = Freq, fill = Var1)) +
  geom_col() +
  coord_polar(theta = "y") +
  xlim(c(0.2, hsize + 0.5))

#Insight : This graph shows how the rating is distributed.

## Get a feel about the distribution of text lengths of the book descriptions by adding a new feature for the length of each message. Check the statistical values.

#Creating a new feature named desc_len
data_goodread$desc_len <- str_count(data_goodread$desc)

#statistical values of the variable
skim(data_goodread)

#Insight : The dataset consist of 5000 observations and 4 features.
#There are two column types. Namely, factor and numeric.
#The distribution of the variables are skewed, where rating and rating category exhibits left skewness and desc_len shows right skewness.

## Visualize the distribution of text lengths with a histogram, where the colouring is according to the rating category

datalist <- data_goodread[c(3, 4)] 
dataFrame <- datalist %>% 
  as_tibble()
dataFrame <- na.omit(dataFrame)
coul <- brewer.pal(5, "Set2") 
barplot(dataFrame$desc_len,
        main = "Description Length ",
        ylab = "",
        xlab = "Length",
        names.arg = dataFrame$rating_category,
        col = coul,
        width=c(1),
        las=1,
        horiz = FALSE
)

## Create a random stratified training and test split (70/30 split). Verify the correct proportions of the splitted data sets by creating proportion table.

set.seed(123)
data_split = sample.split(data_goodread$rating_category, SplitRatio = .7)
train_data = subset(data_goodread, data_split == TRUE)
test_data  = subset(data_goodread, data_split == FALSE)
# check the proportions
prop.table(table(train_data$rating_category))
prop.table(table(train_data$rating_category))

## Tokenize the descriptions with help of the quanteda package and illustrate the effect of each action by showing the content of some of the reviews in the training data set

#Creating corpus for train data:
corpus_train <- VCorpus(VectorSource(train_data$desc))
as.character(corpus_train[[2]])

#Clean the corpus with the help of the tm map() function - transformer()
#1. Casing
corpus_train <- tm_map(corpus_train, content_transformer(tolower))
as.character(corpus_train[[2]])
#2. Puntucation
corpus_train <- tm_map(corpus_train, removePunctuation)
as.character(corpus_train[[2]])
#3. Number
corpus_train <- tm_map(corpus_train, removeNumbers)
as.character(corpus_train[[2]])
#4. Stop words
corpus_train <- tm_map(corpus_train, removeWords, stopwords("english"))
as.character(corpus_train[[2]])
#5. Reducing root words
corpus_train <- tm_map(corpus_train, stemDocument)
as.character(corpus_train[[2]])
#6. Removed white space
corpus_train <- tm_map(corpus_train, stripWhitespace)
as.character(corpus_train[[2]])

## Create a bag-of-words model and add bi-grams to the normal feature matrix.
dtm_train <- DocumentTermMatrix(corpus_train)
dtm_train

#Now the train data is convereted into bag of words where each text is converted as a feature .

inspect(dtm_train)

#Drawing a simple word cloud of the most frequent repeated words
freq_term_cloud = data.frame(sort(colSums(as.matrix(dtm_train)), decreasing=TRUE))
wordcloud(rownames(freq_term_cloud), freq_term_cloud[,1], max.words=100, colors=brewer.pal(1, "Dark2"))

#This graphs displays the frequently repeated words.
#To illustrate New, book, life and one which are most repeated.

## Build a function for relative term frequency (TF) and another one to calculate the inverse document frequency (IDF). Use both functions to create a function which calculated the TF-IDF.Use these three functions on the feature matrix separately to create a term-document frequency matrix.
dtm_train <- DocumentTermMatrix(corpus_train, control = list(weighting = weightTfIdf))

#Insight: Here we can observe that there are several empty documents.

## Look for incomplete cases and fix them if needed
#Removing the empty documents
rowTotals <- slam::row_sums(dtm_train)
dtm_train <- dtm_train[rowTotals > 0, ]

#Removing sparsity
dtm_train <- removeSparseTerms(dtm_train, 0.99)

## Check the dimensionality of the frequency matrix. What problems can you see with the current frequency matrix besides of the "Curse of Dimensionality"
#Converting to matrix
train_data_matrix <- as.matrix(dtm_train)
#Checking dimensionality of the frequency matrix
train_data_matrix[300:310, 1000:1010]

#A problem here is there are more features in the dataset than observations which might lead to a risk of  overfitting of the train dataset.

## Explain what the "Curse of Dimensionality" means in the context of machine learning
#Curse of Dimensionality is a scenario where the machine learning model find different issues in a high dimensionality space which is not discovered in a low dimensionality space. Basically, there are higher number of errors in high dimension space compared to low dimension

## Explain the term Latent Semantic Analysis in the context of NLP
#Latent Semantic Analysis is performed for features that cannot be directly measured.
#It creates representation of text data interm of latent features. It created a Document Term Matrix and Singular Value Decomposition, by reducing the dimensionality of the original dataset by encoding it using the latent features.
#Latent Semantic Analysis is a Natural Language processing technique and an unsupervised learning technique.
#The benefit of this analysis is that it produces the same dimensionality of the original text dataset.

## Reduce the dimensionality of your weighted term document matrix down to 100 columns. Provide summary statistics for the first 10 columns.
train_data_matrix[300:310, 999:1009]

#A term document matrix here shows that the terms is related with the document.
#Here each row defines a term and each column represents a document.










































