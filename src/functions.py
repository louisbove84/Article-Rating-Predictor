import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import string
import nltk
import seaborn as sns
import random
import requests
from math import sqrt
from bs4 import BeautifulSoup
from datetime import datetime
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from collections import Counter
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier,
                              RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier,  XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn import linear_model
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from src.functions import *
from sklearn.metrics import confusion_matrix

class Scraper:
    """Class to scrape www.towardsdatascience.com.

    This class contains all the methods needed to scrape the website:
    www.towardsdatascience.com in order to obtain article information from
    2018-2020. These features include article title and text, number of claps
    received, data, web page link, and responses.

    """

    def __init__(self):
        pass

    #Credit: https://medium.com/the-innovation/scraping-medium-with-python-beautiful-soup-3314f898bbf5
    def convert_day(self, day):

        """This method will take a number between 1 and 365 and return the associated month and day.

        Args:
            day: Number from 1 to 365 indicating which day of the year.

        Returns:
            m: Month (1-12)
            d: Day (1-[28-31])

        """

        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        m = 0
        d = 0
        while day > 0:
            d = day
            day -= month_days[m]
            m += 1
        return (m, d)

    #Credit: https://medium.com/the-innovation/scraping-medium-with-python-beautiful-soup-3314f898bbf5
    def get_claps(self, claps_str):

        """This method will take the string for number of claps an article receives and turn it into a integer.

        Args:
            claps_str: String containing the number of claps found within a web page html

        Returns:
            claps: Integer
        """

        if (claps_str is None) or (claps_str == '') or (claps_str.split is None):
            return 0
        split = claps_str.split('K')
        claps = float(split[0])
        claps = int(claps*1000) if len(split) == 2 else int(claps)
        return claps

    #Credit: https://medium.com/the-innovation/scraping-medium-with-python-beautiful-soup-3314f898bbf5
    def get_medium_data(self):

        """This method will scrape the website: www.towardsdatascience.com
        in order to obtain article information from 2018-2020. These features
        include article title, number of claps received, data, web page link,
        and responses. This function will also save a different CSV file for
        every year scraped.

        Args:
            None

        Returns:
            df_lst: List of data frames from the year 2018 through 2020.

        """

        urls = {'Towards Data Science': 'https://towardsdatascience.com/archive/{0}/{1:02d}/{2:02d}'}
        df_lst = []
        year_lst = [i for i in range(2018,2021)]
        df = pd.DataFrame(columns=['id', 'title', 'subtitle',
                                'claps', 'responses',
                                'publication', 'date'])
        for year in year_lst:
            selected_days = [i for i in range(1, 366)]
            data = []
            article_id = 0
            i = 0 #Counter to help keep track of progress
            n = len(selected_days)
            for d in selected_days:
                i += 1
                month, day = self.convert_day(d)
                date = '{0}-{1:02d}-{2:02d}'.format(year, month, day)
                print(f'{i} / {n} ; {date}')
                for publication, url in urls.items():
                    response = requests.get(url.format(year, month, day), allow_redirects=True)
                    if not response.url.startswith(url.format(year, month, day)):
                        continue
                    page = response.content
                    soup = BeautifulSoup(page, 'html.parser')
                    articles = soup.find_all(
                        "div",
                        class_="postArticle postArticle--short js-postArticle js-trackPostPresentation js-trackPostScrolls")
                    for article in articles:
                        title = article.find("h3", class_="graf--title")
                        if title is None:
                            continue
                        title = title.contents[0]
                        article_id += 1
                        subtitle = article.find("h4", class_="graf--subtitle")
                        subtitle = subtitle.contents[0] if subtitle is not None else ''
                        claps = self.get_claps(article.find_all("button")[1].contents[0])
                        try:
                            link = article.find(class_="postArticle-readMore").find('a').attrs["data-action-value"]
                        except:
                            link = ' '
                        responses = article.find_all("a")
                        if len(responses) == 7:
                            responses = responses[6].contents[0].split(' ')
                            if len(responses) == 0:
                                responses = 0
                            else:
                                responses = responses[0]
                        else:
                            responses = 0

                        data.append([article_id, title, subtitle,
                                    claps, responses,
                                    publication, date, link])

            medium_df = pd.DataFrame(data, columns=['id', 'title', 'subtitle',
                                                    'claps', 'responses',
                                                    'publication', 'date', 'link'])
            medium_df.to_csv('data/medium_data_{}.csv'.format(year))
            df_lst.append(medium_df)
        return df_lst

    def get_medium_text(self, df):

        """This method will scrape the website: www.towardsdatascience.com
        in order to obtain the full article text for all the links within
        the 'link' column of a data frame.

        Args:
            df: Data frame containing web links to all the full text articles
                on www.towardsdatascience.com

        Returns:
            df: Original data frame with an added column containing the scraped
                article text

        """

        urls = df['link']
        df_lst = []
        text_dict = {}

        #Loop through all the popular articles from 2018-2020
        for i in range(len(df['link'])):
            try:
                response = requests.get(df['link'].iloc[i], allow_redirects=True)
                page = response.content
                soup = BeautifulSoup(page, 'html.parser')
                articles = soup.find("p").text
                text = ''
                for a in soup.find_all('p'):
                        text += a.get_text()
                text_dict[i] = text
            except:
                continue

        dict_df = pd.DataFrame.from_dict(text_dict, orient='index', columns=['text'])
        df['text'] = dict_df
        return df


class NLP:
    """Class used to process text data for supervised learning models

    This class contains all the methods needed to process text data in order
    to prepare it for classification and regression models.

    """

    def __init__(self):
        pass

    def extract_ngrams(self, text_lst, num):
        """ Function to generate n-grams from sentences.

        Args:
            text_lst: List of strings
            num: Number of the highest N-Grams appended to the list

        Returns:
            text_lst: List of strings now including the desired N-Grams

        """
        n_grams = ngrams(nltk.word_tokenize(' '.join(text_lst)), num)
        return [ ' '.join(grams) for grams in n_grams]

    def tokenizeTagTrend(self, df_lst, min_word_len=8):
        """This method will process raw sentence strings in preparation
        for wordclouds completing the following:
            -Tokenize (separate words)
            -Lowercase
            -Remove Stopwords
            -Remove Punctuation
            -Tag speech and ONLY keeping nouns
            -Join the remaining words
            -Run a word

        Args:
            df_lst: List of data frames from 2018-2020
            min_word_len: Data frame containing web links to all the full text articles
                on www.towardsdatascience.com

        Returns:
            df: Original data frame with an added column 'change' containing a numeric
            value indicating the increased or decreased use of a certain noun over time

        """

        word_dict = {}
        year = 2018
        punctuation_ = set(string.punctuation)
        stopwords_ = set(stopwords.words('english'))
        stemmer_snowball = SnowballStemmer('english')

        for df in df_lst:
            df['tokenized'] = df['title'].apply(lambda row: word_tokenize(row))
            df['tokenized'] = df['tokenized'].apply(lambda row: [w.lower() for w in row])
            df['tokenized'] = df['tokenized'].apply(lambda row: [w for w in row if not w in punctuation_])
            df['tokenized'] = df['tokenized'].apply(lambda row: [w for w in row if not w in stopwords_])
            #df['tokenized'] = df['tokenized'].apply(lambda row: [stemmer_snowball.stem(w) for w in row]) #Stemming (taking words back to the basics)
            #df['tokenized'] = df['tokenized'].apply(lambda row: extract_ngrams(row,1) + extract_ngrams(row,2)) #Add 2-Grams
            df['tokenized'] = df['tokenized'].apply(lambda row: pos_tag(row))
            word_lst = [a for b in df['tokenized'].tolist() for a in b]
            noun_word_lst = [word for (word, pos) in word_lst if (pos == 'NN' or pos == 'NNS'
                                                                or pos == 'NNP' or pos == 'NNPS')]
            wordcloud = WordCloud(min_word_length = min_word_len).generate(' '.join(noun_word_lst))
            word_dict[year] = pd.Series(wordcloud.words_)
            year += 1

        final_df = pd.DataFrame(word_dict).dropna()
        final_df['change'] =  final_df.iloc[:,2] - final_df.iloc[:, 0]

        return final_df

    def addWordsToCloud(self, dict, lst):
        """ Method to add specific words to wordcloud.

        Args:
            dict: WordCloud dictionary
            lst: List of words to add to wordcloud

        Returns:
            dict: WordCloud dictionary with additional words

        """
        for i in lst:
            dict[i] = np.array(list(dict.values())).mean()
        return dict

    def popular(self, df, pct=0.35):
        """ Method to drop rows that fall inbetween the top and bottom % of total claps.

        Args:
            df: Dataframe to filter
            pct: Percent of total claps user wants to keep

        Returns:
            dict: DataFrame with only top and bottom percent of total claps

        """
        df['popular'] = df['claps'].sort_values().apply(lambda x: 1 if x > df['claps'].quantile(q=(1-pct))
                                                        else (0 if x < df['claps'].quantile(q=pct) else 2))
        return df

    def vectorize(self, df, col_name):
        """ Method to vectorize a column of text

        Args:
            df: Dataframe to vectorize
            col_name: Specify the column name to use

        Returns:
            df_tfidf: Vectorized DataFrame

        """
        vectorizer = TfidfVectorizer(analyzer='word')
        vectors = vectorizer.fit_transform(df[col_name])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df_tfidf = pd.DataFrame(denselist, columns = feature_names)

        return df_tfidf

    def evaluate_reg_models(self, model_lst, X_train, y_train, X_test, y_test):
        """ Method to evaluate different regression models with vectorized text

        Args:
            model_lst: List of models to test
            X_train: X_train data to use for evaluation
            y_train: y_train data to use for evaluation
            X_test: X_test data to use for evaluation
            y_test: y_test data to use for evaluation

        Returns:
            rmse_dict: Dictionary with the models and their RMSE results

        """
        rmse_dict = {}
        r2_dict = {}

        for model in model_lst:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))
            print(f"The RMSE of model {type(model).__name__} is {rmse :.2f}")
            print("\n")

            rmse_dict[type(model).__name__] = rmse

        #Plot the RMSE results
        plt.figure(num=None, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
        plt.barh(*zip(*rmse_dict.items()))
        plt.xlabel('RMSE', size = 25)
        plt.ylabel('Model', size = 25)
        plt.title('Regression Model Comparison', size = 25)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.grid()
        plt.show()

        return rmse_dict

    def evaluate_cla_models(self, model_lst, X_train, y_train, X_test, y_test):
        """ Method to evaluate different classification models with vectorized text

        Args:
            model_lst: List of models to test
            X_train: X_train data to use for evaluation
            y_train: y_train data to use for evaluation
            X_test: X_test data to use for evaluation
            y_test: y_test data to use for evaluation

        Returns:
            cla_results_dict: Dictionary with the models and their F-1 results

        """
        cla_results_dict = {}
        cm_results_dict = {}
        for model in model_lst:
            model.fit(X_train, y_train)
            y_pred= model.predict(X_test)
            accuracy= accuracy_score(y_test, y_pred)
            clf_report= classification_report(y_test, y_pred,output_dict=True)
            print(f"The accuracy of model {type(model).__name__} is {accuracy:.2f}")
            print(f"The Precision of model {type(model).__name__} is {clf_report['macro avg']['precision'] :.2f}")
            print(f"The Recall of model {type(model).__name__} is {clf_report['macro avg']['recall'] :.2f}")
            print(f"The F1-Score of model {type(model).__name__} is {clf_report['macro avg']['f1-score'] :.2f}")
            print("\n")

            #labels = [1,0]
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm)
            plt.title('Confusion matrix of {}\n'.format(type(model).__name__), size = 20)
            fig.colorbar(cax)
            plt.xlabel('Predicted', size = 15)
            plt.ylabel('True', size = 15)
            plt.show()
            cla_results_dict[type(model).__name__] = clf_report['macro avg']['f1-score']

        #Plot the F-1 results
        plt.figure(num=None, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
        plt.barh(*zip(*cla_results_dict.items()))
        plt.xlabel('F1-Score', size = 25)
        plt.ylabel('Model', size = 25)
        plt.title('Classification Model Comparison', size = 25)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.xlim(0,1)
        plt.grid()
        plt.show()

        return cla_results_dict