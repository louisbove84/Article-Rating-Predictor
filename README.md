![](images/tds.png)

# Article Rating Predictor

## Table of Contents

* [General Information](#General-Information)
    * [Exploratory Data Analysis](#Exploratory-Data-Analysis)
    * [Supervised Learning Models](#Supervised-Learning-Models)
    * [Final Results](#Final-Results)
    * [Tools Used](#Tools-Used)
    * [Future Improvements](#Future-Improvements)


## General Information
Towards Data Science Inc. provides a platform for thousands of people to exchange ideas and to expand their understanding of data science. This projects aims combine use supervised and transfer learning to create a model that is able to detect whether a particular data science article will be successful on Towards Data Science Inc.'s website. This will enable businesses or authors to quickly assess whether their time will be effectively used on the idea they have.
_______________________________________________
## Exploratory Data Analysis:

Approximately 35,000 article titles, dates, and 'claps' (similar to 'likes') were scraped from 'www.towardsdatascience.com' from 01Jan2018-30Dec2020. The total articles printed increased from around 5000 in 2018 to around 20000 in 2020, which is approximately 50 articles a week! First, the Term Frequency-Inverse Document Frequency (TF-IDF) was used to find words that increased and decreased the most from 2018-2020. Next, the article titles were labeled as popular if they were in the top 35% of 'claps' for that year and unpopular if they were in the bottom 35% of 'claps' for that year, in order to properly separate the target data for classification. Finally, the full article texts were scraped on the popular and unpopular data sets in order to give more fidelity to the results. The word clouds below show terms that increased (in green) and decreased (in red) between 2018 and 2020:

<p align="center" >
  <img src="images/wordcloud.png" width="800">
</p>

### The project files are organized as follows:

- EDA.ipynb: File used to scrape, explore, and transform the data for modeling
- NLP.ipynb: File used for supervised learning models
- images: All images used in the README.md file
- src: Contains a python file with all the functions used in this project
- data: Includes sample '.csv' data to run the EDA & NLP files

### Articles used for help:

* Text Classification with XLNet in Action: https://medium.com/@yingbiao/text-classification-with-xlnet-in-action-869029246f7e
* Scraping Medium with Python & Beautiful Soup: https://medium.com/the-innovation/scraping-medium-with-python-beautiful-soup-3314f898bbf5
____________________________________________________________

## Supervised Learning Models:

***Step 1: Establish Training and Testing Data***

The training and testing data was compiled from articles between 2018 to 2020 and split into a data set containing the top and bottom 35% of articles based on the number of 'claps' they received. Examples from the popular and unpopular article titles are seen below:

<p align="center" >
  <img src="images/title_examples.png" width="800">
</p>

***Step 2: Classification on Article Titles***

A list of classification models were used on the popular/unpopular article titles from the data set including: Decision Tree, Random Forest, KNN, XGBClassifier, and Gradient Boosting Classifier. The figure below shows how poorly these models did using only the article titles. The best model was the Random Forrest Classifier with an F-1 score of 0.66.

<p align="center" >
  <img src="images/titles_classification.png">
</p>

***Step 3: Regression on Article Titles***

Due to the inaccuracies from the classification results the decision was made to run a list of regression models on article titles from the entire data set using the number of 'claps' as the target data. The results below do not show much improvement from the baseline model with an RMSE of 1907. The best model in comparison is the Ridge Regression with an RMSE of 1886.


<p align="center">
  <img src="images/titles_regression.png">
</p>

***Step 4: Classification on Full Article Text***

In order to improve accuracy the full article text was used with the same classification labels used in Step 2. The results improved slightly with the Gradient Boosting Classifier achieving an F-1 score of 0.72.

<p align="center">
  <img src="images/text_classification.png">
</p>

***Step 5: Resgression on Full Article Text***

Finally, the same full article text was used with the same regression information from Step 3. The results are shown below with none of the models doing better than the baseline model RMSE of 1907.

<p align="center">
  <img src="images/text_regression.png">
</p>
________________________________

## Final Results

The final results from the supervised models shows that the insufficient ... model produced the best results in order to predict the popularity of an article based on its title.

 Confusion matrices from the base and final model are shown below for comparison.

<p align="center">
  <img src="images/results.png">
</p>
_______________________________________

## Tools Used

***Python:***

Data Gathering: Pandas<br>
Data Analysis: Google Colab, Tensor Flow, Pandas, Scikit-Learn, NLTK<br>

***Visualization:***

Data Visualization: Matplotlib

_______________________________________
## Future Improvements

1. Increase accuracy of the models by running more full text articles through the models.
2. Scrape more articles from other websites.
