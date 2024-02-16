
# -*- coding: utf-8 -*-


import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
#from ctransformers.langchain import CTransformers
from langchain_community.llms import CTransformers

def getLlamaResponse(input_texts, no_words, profession):
    
    llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin', model_type='llama',
                        config = {'max_new_tokens': 256,
                                  'temperature': 0.01} )    
    print(llm('AI is going to'))
    template = """
                Write a article for {profession} job profile for a topic {input_texts} 
                within {no_words} words
                """
    prompt = PromptTemplate(input_variables=["profession","input_texts",'no_words'],
                            template = template)

    
    ## Generate the response form the LLAMA 2 model

    response = llm(prompt.format(profession=profession,input_texts=input_texts,no_words=no_words))
    print("LLAMA model response :\n", response)
    return response
    
    
st.set_page_config(page_title = "GenAI Search Result",
                   page_icon = "None",
                   layout = 'centered',
                   initial_sidebar_state= 'collapsed'
        )


st.header("GenAI Search Result:")

input_texts = st.text_input("Enter the search topic")

## creating 2 more columns for additional field
col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No of words')
with col2:
    profession = st.selectbox('Writing the articls for', 
                             ('Researcher', 'Data Scientist','Common People'), index = 0)
    
submit = st.button("Generate")

# Final response
if submit:
    st.write(getLlamaResponse(input_texts, no_words, profession))

    
