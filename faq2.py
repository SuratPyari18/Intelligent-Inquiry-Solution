from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore()
from haystack.nodes import EmbeddingRetriever
retriever = EmbeddingRetriever(
document_store=document_store,
embedding_model="sentence-transformers/all-MiniLM-L6-v2",
use_gpu=False,
scale_score=False,
)

import pandas as pd
df = pd.read_excel(r'C:\Users\HP\OneDrive\Desktop\project1\CHATBOT_FAQ.xlsx')

questions = list(df["question"].values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"question": "content"})

docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

from haystack.pipelines import FAQPipeline
pipe = FAQPipeline(retriever=retriever)
from haystack.utils import print_answers

#UI
import streamlit as st
from translate import Translator
from langdetect import detect

# Function to translate text to Hindi
def translate_to_hindi(text):
    translator = Translator(to_lang='hi')
    translation = translator.translate(text)
    return translation

def translate_to_english(text):
    translator = Translator(to_lang='en')
    translation = translator.translate(text)
    return translation

language = st.selectbox("Select Language", ["English", "Hindi"])
if language == "English":
    st.title("Question Answering system")
    question = st.text_area("Enter your question:")
    if st.button("Answer"):
        params={"Retriever": {"top_k": 1}}
        prediction = pipe.run(query=question, params=params)
        for ans in prediction['answers']:
            st.write(ans.answer)
            st.write('---')

elif language == "Hindi":
    st.title("प्रश्नोत्तरी प्रणाली")
    question = st.text_area("अपना प्रश्न दर्ज करें:")
    if st.button("उत्तर दें"):
        detected_lang = detect(question)
        if detected_lang != 'hi':
            question = translate_to_hindi(question)
        params={"Retriever": {"top_k": 1}}
        prediction = pipe.run(query=question, params=params)
        for ans in prediction['answers']:
            answer = ans.answer
            if detected_lang != 'hi':
                answer = translate_to_english(answer)
            st.write(answer)
            st.write('---')
