import  pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import time
import numpy as np
import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
print(os.getcwd())
import warnings
warnings.filterwarnings("ignore")
usr_name="datascience"
password="prediction"




def Load_dataset():
    print("list of datasets available")
    print("AUSTRALIA's Sydney city WEATHER DATA")
    s=input(" do you want to load the data set :yes/no")
    if(s=='yes'):
     raw_dataset=pd.read_csv(r'sydney.csv')
     #test_data=raw_dataset[raw_dataset[] & raw_dataset[]]

     return raw_dataset
    else:
        print("Add another dataset and restart the program ")
        import sys
        sys.exit()



def preprocessing(raw_dataset):
     print("now preprocessing the existing data set")
     print("step 1: let us see the the null value percentage for each coloumn in the data set")
     n_colomns_list=raw_dataset.isnull().mean()
     #print(n_colomns_list)
     #print(dict(n_colomns_list))
     x=dict(n_colomns_list)
     #print("for the better model training we consider the feautures which are having atleast 70 percent of data ")
     #print("omitting the featues having more than")
     filtered_list=[]
     for i in x:
        if x[i]<0.3:
            print(i,x[i])
            filtered_list.append(i)
     filter_2_df=raw_dataset[ list(filtered_list)]
     # EXTRACTING YEAR FROM DATE
     filter_2_df['year'] = pd.DatetimeIndex(filter_2_df['Date']).year
     # filter_2_df['month']=pd.DatetimeIndex(filter_2_df['Date']).month
     '''
     print(filter_2_df['year'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                                title=' DISTRUBUTION OF RAINFALL DATA  OVER YEARS ',
                                                                color='cyan'))

    
    filter_2_df=filter_2_df.replace(['NA'],np.nan)
     filter_2_df = filter_2_df.replace(' ', np.nan)
     '''
     print("checking the null mean score before replacing them")
     #print(filter_2_df.isnull().mean())
     #print(filter_2_df.dtypes)

     print("if it is categorical data we will replace with mode ,if its is numerical data we will replace null values with mean")

     #code for replacing null values

     for i in filter_2_df :

          if(filter_2_df[i].dtypes == 'float64'):
              #print(i, filter_2_df[i].dtypes)
              filter_2_df[i] = filter_2_df[i].fillna(filter_2_df[i].mean())
              #filter_2_df[i].value_counts()



          if(filter_2_df[i].dtype =='object'):
              #print(i, filter_2_df[i].dtypes)
              #filter_2_df[i]=filter_2_df[i].astype('str')

              #print(filter_2_df[i].value_counts())
              mode_val = filter_2_df[i].mode()
              filter_2_df[i].replace(np.nan,filter_2_df[i].mode(),inplace=True)
              filter_2_df[i].fillna(filter_2_df[i].mode(),inplace=True)
              filter_2_df[i].replace("nan", filter_2_df[i].mode(), inplace=True)
              filter_2_df[i].replace("NaN", filter_2_df[i].mode(), inplace=True)



              #print("valuecounts after replcing", filter_2_df[i].value_counts(dropna=False))
              #print("count of mode after repacing", filter_2_df[i].mode().count())


     print("now all the missing values are handled lets check them")

     #filter_2_df['month']=pd.DatetimeIndex(filter_2_df['Date']).month
     #filter_2_df.groupby('Location')['year'].value_counts().sort_index().plot(kind='bar',figsize=(20,15),title=' DISTRUBUTION OF RAINFALL DATA  OVER YEARS ',color='cyan')
     #as we recorded year


     #plt.show()
     for i in filter_2_df:

     #print(filter_2_df[i].value_counts())
     #now feature scaling and label encoding the datasset

        if (filter_2_df[i].dtypes == 'float64' and i not in ['year'] ):

          mean=filter_2_df[i].mean()
          print("mean of the colomn",i,mean)
          stdeviation=filter_2_df[i].std()
          print("standard deviation of ",i,stdeviation)

          filter_2_df[i]= filter_2_df[i].apply(lambda x:((x-mean)/stdeviation))
          #print("scaled values  after standardization are ",filter_2_df[i].value_counts())
     #filter_2_df.to_csv(r'pre@'+str(datetime.time())+'.csv')

        filter_2_df[i].dtypes == 'float64'


     filter_2_df=filter_2_df.drop(columns=['Date'])
     print("scaling done  now label encoding the dataset")
     filter_2_df["RainToday"] = filter_2_df["RainToday"].map({"No": 0, "Yes": 1})
     filter_2_df["RainTomorrow"] = filter_2_df["RainTomorrow"].map({"No": 0, "Yes": 1})

     for i in filter_2_df:

         if (filter_2_df[i].dtypes == 'object'  and i not in ['RainToday','RainTomorrow']):
             print("converting the other categorical colomns to   machine encoded form")
             filter_2_df[i]=pd.get_dummies(filter_2_df[i])
             print("coloumn ",i,filter_2_df[i].value_counts())






     return filter_2_df







def feature_selection(df):

    print("selecting the sensitive feautures with respect to target  based on correlations in the data")
    #from sklearn.ensemble import ExtraTreesClassifier
    #print(df["RainToday"].value_counts())
    #print(df["RainToday"].mode())
    #df["RainToday"]=df["RainToday"]
    df["RainToday"].fillna(0,inplace=True)
    #df["RainToday"] = df["RainToday"].replace(np.nan, df["RainToday"].mode())

    #print(df["RainToday"].value_counts(dropna=False))
    #print(df["RainToday"].mode())
    df["RainToday"].replace(np.nan, df["RainToday"].mode(), inplace=True)
    df["RainToday"].replace("nan", df['RainToday'].mode(), inplace=True)
    #df["RainToday"].replace("NaN", df['RainToday'].mode(), inplace=True)
    print("filling missing values and scaling of needed colomns are done lets check them finally ")
    #print(df["RainToday"].value_counts(dropna=False))

    file='@completly_preprocessed_data.csv'
    df.to_csv(file)
    print("checking missing valued colmns:",df.isnull().any())

    corr=df.corr()
    sensible_cols=dict(corr['RainTomorrow'])

    listn=[]
    print(sensible_cols)
    for i in sensible_cols:
        if(sensible_cols[i]>0.2):
            listn.append(i)
    print(listn)

    print("the colomns which are effecting more than 20% are ",listn)
    print("now we are selecting these colomns to predict the no of raiing days across australia                 ")
    #corr.to_csv('correlations.csv')
    #print(df['Location'].value_counts())
    df_train=pd.DataFrame()
    df_test=pd.DataFrame()

    test_year=int(input("select the year to predict raniny days acorss the country from 2016 and 2017 "))
    df_test=df[df['year']==test_year]
    df_test=df_test[listn]
    df_test.to_csv(r'test_data.csv')
    df_train = df[df['year'] != test_year]
    df_train = df_train[listn]
    df_train.to_csv(r'train_data.csv')
    print("length pf train,test",len(df_train),len(df_test))
    import seaborn as sns
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True)
    plt.show()
    return df_train,df_test



def model_slection(df,df_test):
    print("modelling now")
    dft=df_test
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    '''
    X=df[['Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'RainToday', 'RISK_MM',]]
    Y=df[['RainTomorrow']]
    '''
    X = df.drop(columns=['RainTomorrow'])

    Y = df['RainTomorrow']


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    # making predictions on the testing set
    y_pred = gnb.predict(x_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    from sklearn import metrics
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

    from sklearn import svm

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    # print(x_train.value_counts())
    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print("supprot vector classcification model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    opt=int(input("select 1 for naive bayes ,2 for SVM "))
    if(opt==1):
      dft=df_test.drop(columns=['RainTomorrow'])
      a = len(dft)
      print(" the total  number of records for all the cities accoross the australia for the given year are", a)

      pred= gnb.predict(dft)
      dft['predictions']=list(pred)
      print(" predictions of   no of rainy and non rainy days for the given are 0 = non rainy 1= rainy days ")
      print(dft['predictions'].value_counts())
      dft.to_csv(r'predictions.csv')
      dft['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                              title='Naive bayes  prediction distrubution in test data',
                                                              color='cyan')
      plt.xlabel("PREDICTED CLASSES")
      plt.ylabel('COUNT')
      plt.show()
    elif(opt==2):
        dft = df_test.drop(columns=['RainTomorrow'])
        a =len(dft)
        print(" the total  number of records for all the cities accoross the australia for the given year are",a)
        pred = clf.predict(dft)
        dft['predictions'] = list(pred)
        print(" predictions of   no of rainy and non rainy days for the given are 0 = non rainy 1= rainy days ")
        print(dft['predictions'].value_counts())
        dft.to_csv(r'predictions.csv')
        dft['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                            title='SVM  prediction distrubution in test data',
                                                            color='cyan')
        plt.xlabel("PREDICTED CLASSES")
        plt.ylabel('COUNT')
        plt.show()
    else:
     print(" please give the numbers only in the given list")

if(__name__ == "__main__") :
    print(" ENTER THE LOGIN CREDINTIALS TO ACCESS THE RAIFALL PREDICTION ")
    u=input("ENTER USER NAME ..: ")
    p=input("ENTER PASSWORD..:  ")
    if(u==usr_name and p==password):
      #loading the dataset Load_dataset function this function loads and returns existing dataset

      raw_dataset=Load_dataset()
        #raw_dataset['year']
      #preprocessing the dataset
      #prepocessing function this function takes our rawdataset and processes to
      pre_processed=preprocessing(raw_dataset)

      '''
      print(pre_processed.isnull().any())
      for i in pre_processed:
          print(pre_processed[i].value_counts(dropna=False))
       
     print(pre_processed.dtypes)
      '''
      df_train, df_test= feature_selection(pre_processed)
      model_slection(df_train,df_test)


    else:
         print("CREDENTIALS ARE NOT MATCHING ,PLEASE RETRY WITH VALID USERNAME AND PASSWORD")











