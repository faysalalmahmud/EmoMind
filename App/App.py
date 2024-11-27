#Core Package
# from random import choice

import streamlit as st
import plotly.express as px


#EDA Package
import pandas as pd
import numpy as np

#Utils
import joblib
from streamlit import image

# from streamlit import columns


#Function Call
pipe_lr = joblib.load(open("Model/emotion_classifier_pipe_lr_2.pkl", "rb"))

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠", "disgust" : "🤮", "fear" : "😨", "happy" : "😊", "joy" : "😃", "neutral" : "😐", "sadness" : "🥹", "shame" : "🫢", "surprise" : "😮"}



def main():
    # st.title("Emotion Detection App")
    st.markdown("<h1 style='text-align: center; color: black;'>EmoMind</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: gray;'>An Emotion Detection App</h5>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: gray;'>{ Developed By TEAM MUSKETEER }</h6>", unsafe_allow_html=True)
    # st.text("Developed By Team Musketeer")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    st.sidebar.markdown('<div style="text-align: center; position: fixed; bottom: 0; font-size:15px;">Developed with ❤️ by <br> TEAM MUSKETEER</div>', unsafe_allow_html=True)

    if choice == "Home":
        # st.snow()
        # st.subheader("Home \nEmotion In Text")
        st.divider()
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Enter your text", key='emotion_clf_text')
            submit_text = st.form_submit_button("Submit")
            st.markdown("<p style='text-align: center; color: gray; font-size:12px'>EmoMind can make mistakes, so double-check it</p>", unsafe_allow_html=True)
        
        if submit_text and raw_text != '':
            col1, col2 = st.columns(2)
            
            #Apply Function Here
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:

                with st.container(border=2):
                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                with st.container(border=2):
                    st.success("Predicted Probability")
                    # st.write(probability)
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    # st.write(proba_df.T)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["Emotions", "Probability"]
                    # fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability')
                    fig = px.bar(proba_df_clean, x='Emotions', y='Probability')
                    st.plotly_chart(fig, use_container_width=True)


    # elif choice == "Monitor":
    #     st.subheader("Monitor App")
    else:
        st.subheader("About")



if __name__ == '__main__':
    main()

