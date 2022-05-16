# NOTE: 
    
# Deployed this app without balance our dataset 

# USE: LogisticRegression model


import pickle
import streamlit as st


logi=pickle.load(open("app1_nlp_deploy_model.pkl","rb"))

def predict_emotions(docx):
      result=logi.predict([docx])
      if result==1:
          return "Positive"
      elif result==-1:
          return "Negative"
      else:
          return "Neutral"

def get_prediction_proba(docx):
    result=logi.predict_proba([docx])
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
    
        
    
