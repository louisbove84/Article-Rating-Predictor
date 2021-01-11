class Scraper:
    """Class to scrape www.towardsdatascience.com.

    This class contains all the methods needed to scrape the website:
    www.towardsdatascience.com in order to obtain article information from
    2018-2020. These features include article title and text, number of claps
    recieved, data, webpage link, and responses.

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

        """This method will take the string for number of claps an article recieves and turn it into a integer.

        Args:
            claps_str: String containing the number of claps found within a webpage html

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
        include article title, number of claps recieved, data, webpage link,
        and responses.

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

    #Scrape entire article text from TowardsDataScience.com
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
    """Class to scrape www.towardsdatascience.com.

    This class contains all the methods needed to scrape the website:
    www.towardsdatascience.com in order to obtain article information from
    2018-2020. These features include article title and text, number of claps
    recieved, data, webpage link, and responses.

    """

    def __init__(self):
        pass

    # Function to generate n-grams from sentences.
    def extract_ngrams(self, text_lst, num):
        n_grams = ngrams(nltk.word_tokenize(' '.join(text_lst)), num)
        return [ ' '.join(grams) for grams in n_grams]

    #Tokenize, Lowercase, and Filter Stopwords for Column in DataFrame
    def tokenizeTagTrend(self, df_lst, min_word_len=8):

        word_dict = {}
        year = 2018

        #Pull down stopwords and punctuation
        punctuation_ = set(string.punctuation)
        stopwords_ = set(stopwords.words('english'))

        #Use Snowball Stemmer for stemming
        stemmer_snowball = SnowballStemmer('english')

        #Run through all the Data Frames
        for df in df_lst:

            #Modify text
            df['tokenized'] = df['title'].apply(lambda row: word_tokenize(row)) #Tokenize (separate words)
            df['tokenized'] = df['tokenized'].apply(lambda row: [w.lower() for w in row]) #Lowercase
            df['tokenized'] = df['tokenized'].apply(lambda row: [w for w in row if not w in punctuation_]) #Remove Stopwords
            df['tokenized'] = df['tokenized'].apply(lambda row: [w for w in row if not w in stopwords_]) #Remove Punctuation
            #df['tokenized'] = df['tokenized'].apply(lambda row: [stemmer_snowball.stem(w) for w in row]) #Stemming (taking words back to the basics)
            #df['tokenized'] = df['tokenized'].apply(lambda row: extract_ngrams(row,1) + extract_ngrams(row,2)) #Add 2-Grams
            df['tokenized'] = df['tokenized'].apply(lambda row: pos_tag(row)) #Tag speech with noun/adj/verb/etc.
            word_lst = [a for b in df['tokenized'].tolist() for a in b] #Combine the tokenized column to a list
            noun_word_lst = [word for (word, pos) in word_lst if (pos == 'NN' or pos == 'NNS' #Only keep nouns
                                                                or pos == 'NNP' or pos == 'NNPS')]
            wordcloud = WordCloud(min_word_length = min_word_len).generate(' '.join(noun_word_lst)) #Wordcloud
            word_dict[year] = pd.Series(wordcloud.words_) #Take the word frequencies
            year += 1

        final_df = pd.DataFrame(word_dict).dropna()
        final_df['change'] =  final_df.iloc[:,2] - final_df.iloc[:, 0]

        return final_df

    #Tokenize, Lowercase, Filter Stopwords, and Tag Nouns for Column in DataFrame
    def tokenizeTag(self, df_lst):

        word_dict = {}
        year = 2018

        #Pull down stopwords and punctuation
        punctuation_ = set(string.punctuation)
        stopwords_ = set(stopwords.words('english'))

        #Use Snowball Stemmer for stemming
        stemmer_snowball = SnowballStemmer('english')

        #Run through all the Data Frames
        for df in df_lst:

            #Modify text
            df['tokenized'] = df['text'].apply(lambda row: word_tokenize(row)) #Tokenize (separate words)
            df['tokenized'] = df['tokenized'].apply(lambda row: [w.lower() for w in row]) #Lowercase
            df['tokenized'] = df['tokenized'].apply(lambda row: [w for w in row if not w in punctuation_]) #Remove Stopwords
            df['tokenized'] = df['tokenized'].apply(lambda row: [w for w in row if not w in stopwords_]) #Remove Punctuation
            df['tokenized'] = df['tokenized'].apply(lambda row: pos_tag(row)) #Tag speech with noun/adj/verb/etc.
            df['tokenized'] = df['tokenized'].apply(lambda x: x[0] if (x[1] == 'NN') or (x[1] == 'NNS')
                                                    or (x[1] == 'NNP') or (x[1] == 'NNPS') else '')
                                                    #Only accept Nouns
        return df

    #Create popular column for those articles in the upper and lower 35% range of 'claps' (basically 'likes')
    def popular(self, df):
        df['popular'] = df['claps'].sort_values().apply(lambda x: 1 if x > df['claps'].quantile(q=0.65)
                                                        else (0 if x < df['claps'].quantile(q=0.35) else 2))
        return df

    #Vectorize bag-of-words using TFIDF
    def vectorize(self, df, col_name):

        vectorizer = TfidfVectorizer(analyzer='word')
        vectors = vectorizer.fit_transform(df[col_name])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df_tfidf = pd.DataFrame(denselist, columns = feature_names)

        return df_tfidf

    #Evaluate all the regression models
    def evaluate_reg_models(self, reg_model_lst):

        rmse_dict = {}
        r2_dict = {}

        for model in models:
            model.fit(X_train, y_train) # fit the model
            y_pred = model.predict(X_test) # predict on the test set
            rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))
            score = metrics.r2_score(y_test, y_pred)
            print(f"The RMSE of model {type(model).__name__} is {rmse :.2f}")
            print("\n")

            rmse_dict[type(model).__name__] = mse

        #Plot the F-1 results
        plt.bar(*zip(*rmse_dict.items()))
        plt.show()

        return rmse_dict

    #Evaluate all the classification models
    def evaluate_cla_models(self, reg_model_lst):

        cla_results_dict = {}

        for model in models:
            model.fit(X_train, y_train) # fit the model
            y_pred= model.predict(X_test) # predict on the test set
            accuracy= accuracy_score(y_test, y_pred) # model accuracy
            clf_report= classification_report(y_test, y_pred,output_dict=True) # precision and recall
            print(f"The accuracy of model {type(model).__name__} is {accuracy:.2f}")
            print(f"The Precision of model {type(model).__name__} is {clf_report['macro avg']['precision'] :.2f}")
            print(f"The Recall of model {type(model).__name__} is {clf_report['macro avg']['recall'] :.2f}")
            print(f"The F1-Score of model {type(model).__name__} is {clf_report['macro avg']['f1-score'] :.2f}")
            print("\n")

            cla_results_dict[type(model).__name__] = clf_report['macro avg']['f1-score']

        #Plot the F-1 results
        plt.bar(*zip(*results_dict.items()))
        plt.show()

        return cla_results_dict