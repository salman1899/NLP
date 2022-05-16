# Here Balanced Data we have,
# Using RandomForest Model,

import pickle
import streamlit as st
import pandas as pd 
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#rf=pickle.load(open("nlp_salman.pkl","rb"))
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
lm=WordNetLemmatizer()
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))# removing charaters apart from alphabets
        new_item = new_item.lower() # make it lowercase
        new_item = new_item.split() # instead of using tokanization we can use split as well
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        # removing stopwords and perform lemmatization
        corpus.append(' '.join(str(x) for x in new_item)) # finally join the strings(reviews)
    return corpus
df=pd.read_csv("deploy_nlp_data.csv")
_a=text_transformation(df["body"])
cvv=CountVectorizer()
c=cvv.fit_transform(_a)
x=c
y=df["label"]
#balancing data
from imblearn.over_sampling import SMOTE
oversampling=SMOTE()
x,y=oversampling.fit_resample(x,y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)
model=RandomForestClassifier()
model.fit(xtrain,ytrain)

def predict_emotions(docx):
      a=text_transformation([docx])
      b=cvv.transform(a)
      result=model.predict(b)
      if result==1:
          return "Positive"
      elif result==-1:
          return "Negative"
      else:
          return "Neutral"

def get_prediction_proba(docx):
    a=text_transformation([docx])
    b=cvv.transform(a)
    result=model.predict_proba(b)
    return result
        
def main():
   st.title("Emotion Classifier App")
   menu=["Home","Help","About"]
   choice=st.sidebar.selectbox("Menu",menu)    
   
   if choice == "Home":
       st.subheader("Home")
        
       
       with st.form(key="emotion_clf_form"):   
          raw_text = st.text_area("Type Here")
          submit_text=st.form_submit_button(label="submit")


       if submit_text:  
           col1,col2 = st.columns(2)
           
           predictions=predict_emotions(raw_text)

           predictions_proba=get_prediction_proba(raw_text)
           with col1: 
            st.success("Prediction")
            st.write(predictions)
            st.success("Prediction Proba")
            st.write(predictions_proba)
           
            
           with col2: 
            st.success("Original Text")
            st.write(raw_text)   
           
            
   elif choice == "Help":
        st.subheader("Help")
        
   else:   
        st.subheader("About")




if __name__ == '__main__':
    main()
    






    
        
    
