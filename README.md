![](Images/facial_recognition_action.jpg)

# Medium Article Generator

## Table of Contents

* [General Information](#General-Information)
    * [Exploratory Data Analysis](#Exploratory-Data-Analysis)
    * [GPT-2](#GPT2)
    * [Final Results](#Final-Results)
    * [Tools Used](#Tools-Used)
    * [Future Improvements](#Future-Improvements)


## General Information
Towards Data Science Inc. provides a platform for thousands of people to exchange ideas and to expand their understanding of data science. This projects aims combine use supervised and transfer learning to create a model that is able to detect popular data science article subjects and create an article based on that title. This model will allow someone that sees a Data Science topic in either a news article or web page, to quickly assess if it would be popular on 'Towards Data Science' and then generate the rough draft of an article. Businesses could use models like this to speed up production of articles or news stories.
_______________________________________________
## Exploratory Data Analysis:

Approximately 35,000 article titles, dates, and 'claps' (similar to 'likes') were scraped from 'www.towardsdatascience.com' from 01Jan2018-30Dec2020. The total articles printed increased from around 5000 in 2018 to around 20000 in 2020, which is approximately 50 articles a week! Term Frequency-Inverse Document Frequency (TF-IDF) was used to find words that increased and decreased the most from 2018-2020. Finally, article titles were labelled as popular if they were in the top 35% of 'claps' for that year in order to find the most popular articles to feed into the GPT-2 model. The word clouds below show terms that increased (in green) and decreased (in red) between 2018 and 2020:

<p align="center" >
  <img src="images/wordcloud.png" width="800">
</p>

### The project files are organized as follows:

- EDA.ipynb:
- NLP.ipynb: Project executable code used in AWS SageMaker
- text: Includes the compiled text files used for fine tuning the CPT-2 model
- src: Includes all the functions used in the EDA & NLP files
- data: Includes all the csv files created from the EDA & NLP files

### Articles used for help:

>2

____________________________________________________________

## Supervised Learning Models:


***Step 1: Establish Training and Testing Data***

The training and testing data was compiled from articles between 2018 to 2020. The plot below shows the distribution of ages from the second data set.

<p align="center" >
  <img src="images/model2ages.png" width="800">
</p>

Due to the uneven distribution of data the decision was make to split the final dataset into 8 different age groups (0-3, 4-7, 8-14, 15-24, 25-37, 38-47, 48-59, 60-) in order to more evenly distribute the data, as seen below.

<p align="center" >
  <img src="Images/age_breakdown.png" width="800">
</p>

***Step 2: Base Model Results***

The initial model used a combination of Convolutional, Max Pooling, Dense, and Dropout Layers with a test accuracy of ***61%***. Seen in the confusion matrix below, the most difficult ages to predict corresponded to the age ranges with the least amount of images: 4-7, 8-14, 38-47, and 48-59

<p align="center" >
  <img src="Images/cm_base_refined.png">
</p>

***Step 3: Improving CNN Model***

Using a tuner function the model was incrementally improved by testing variations in the following areas. The resulting prediction accuracy increased to ***63.4%***.
- 1st Convolutional Layer: Filters - [***32***,64,128,256] = 0.532 or 4% incr.
- 2nd Convolutional Layer: Filters -      [32,64,***128***,256] = 0.552 or 8% incr.
- Dense Layer: Dimensionality Output -    [64,128,512,1024,***2048***] = 0.6006 or 18% incr.
- Dropout: Dropout Rate -                 [***0.2***, 0.4, 0.5, 0.6, 0.8] = 0.5790 or 14% incr.


<p align="center">
  <img src="Images/TunedModelCM.png">
</p>

***Step 4: Image Augmentation***

Using image augmentation the existing combined data set was increased by rotating, shifting, flipping, and shearing images within the original data set.

After running the tuned model over the augmented data set the resulting prediction accuracy increased to 65%. The corresponding confusion matrix is plotted below.

<p align="center">
  <img src="Images/AugModelCM.png">
</p>

________________________________
## Final Results

The final CNN model is a result of hyperparameter tuning of the base model and data set expansion utilizing image augmentation. The final model increased the test accuracy from ***51%*** to ***65%***. Confusion matrices from the base and final model are shown below for comparison.

<p align="center">
  <img src="Images/Model_compare.png">
</p>

_______________________________________
## Tools Used

***Database:***

Data Storage: AWS S3<br>

***Python:***

Data Gathering: Pandas<br>
Data Analysis: AWS Sage Maker, Tensor Flow, Keras, Pandas, Scikit-Learn<br>

***Visualization:***

Data Visualization: Matplotlib

_______________________________________
## Future Improvements

1. Incorporate video capture in order to predict age in real time utilizing webcam feed.
2. Explore the individual age data sets and compare the prediction accuracy to the current model.
