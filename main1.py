from pandas._config.config import options
import streamlit as st
import pandas as pd
import numpy as np
import os 
import sys
from io import BytesIO, StringIO

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer

import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
#Docx resume
import docx2txt
#Wordcloud
import re
import operator
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

set(stopwords.words('english'))
from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd
import docx2txt
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber



STYLE = """
<style>
img{
    max-width:100%;
}
</style>
"""

def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()

	return all_page_text

def read_pdf_with_pdfplumber(file):
	with pdfplumber.open(file) as pdf:
	    page = pdf.pages[0]
	    return page.extract_text()

# import fitz  # this is pymupdf

# def read_pdf_with_fitz(file):
# 	with fitz.open(file) as doc:
# 		text = ""
# 		for page in doc:
# 			text += page.getText()
# 		return text 

# Fxn

st.markdown(

    '''
    <style>
    .main{
        background-color: #ffffff;
    }
    </style>
    ''',
    unsafe_allow_html= True
)


def main():

    st.title(" ------------>Jobshie<--------------")
    st.subheader("Copy the full job description and paste it here:")
    sel_col, disp_col = st.columns(2)
    job_description = sel_col.text_input('','Copy the full job description and paste it here')                                                                                                                ')


    st.subheader("Upload your resume or CV down below:")
    docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
    if st.button("-------------->SHOW ME THE RESULTS<-----------------"):
        if docx_file is not None:
            file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
            #st.write(file_details)
            # Check File Type
            if docx_file.type == "application/octet-stream":
            # Use the right file processor ( Docx,Docx2Text,etc)
                resume = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
                #st.write(resume)


            elif docx_file.type == "application/pdf":
                # raw_text = read_pdf(docx_file)
                # st.write(raw_text)
                try:
                    with pdfplumber.open(docx_file) as pdf:
                        page = pdf.pages[0]
                        resume = page.extract_text()
                        #st.write(page.extract_text())
                except:
                    st.write("None")
                    
                
            elif docx_file.type == "text/plain":
                # raw_text = docx_file.read() # read as bytes
                # st.write(raw_text)
                # st.text(raw_text) # fails
                st.text(str(docx_file.read(),"utf-8")) # empty
                resume = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
                # st.text(raw_text) # Works
                #st.write(resume) # works



            def create_word_cloud(jd):
                corpus = jd
                fdist = FreqDist(corpus)
                #print(fdist.most_common(100))
                words = ' '.join(corpus)
                words = words.split()
                    
                    # create a empty dictionary  
                data = dict() 
                    #  Get frequency for each words where word is the key and the count is the value  
                for word in (words):     
                    word = word.lower()     
                    data[word] = data.get(word, 0) + 1 
                # Sort the dictionary in reverse order to print first the most used terms
                dict(sorted(data.items(), key=operator.itemgetter(1),reverse=True)) 
                word_cloud = WordCloud(width = 800, height = 800, 
                background_color ='white',max_words = 500) 
                word_cloud.generate_from_frequencies(data) 
                
                # plot the WordCloud image
                #plt.figure(figsize = (10, 8), edgecolor = 'k')
                #plt.imshow(word_cloud,interpolation = 'bilinear')
                #plt.axis("off")  
                #plt.tight_layout(pad = 0)
                #plt.show()

            def tokenizer(text):
                ''' a function to create a word cloud based on the input text parameter'''
                ## Clean the Text
                # Lower
                clean_text = text.lower()
                # remove punctuation
                clean_text = re.sub(r'[^\w\s]', '', clean_text)
                # remove trailing spaces
                clean_text = clean_text.strip()
                # remove numbers
                clean_text = re.sub('[0-9]+', '', clean_text)
                # tokenize 
                clean_text = word_tokenize(clean_text)
                # remove stop words
                stop = stopwords.words('english')
                clean_text = [w for w in clean_text if not w in stop] 
                return(clean_text)


            job_description = tokenizer(job_description)
            resume = tokenizer(resume)
            

            resume1 = ' '.join(resume)

            job_description1 = ' '.join(job_description)



            text = [resume1,job_description1]

            cv = CountVectorizer()
            count_matrix = cv.fit_transform(text)


            

            matched_percentage = cosine_similarity(count_matrix)[0][1]*100
            matched_percentage = round(matched_percentage,2)
            mached = "Your resume is " + str(matched_percentage) + "% mached with the job description."
            
            st.subheader(mached)
	    

            text1 = job_description1
            # Create some sample text
            text2 = resume1

            # Create and generate a word cloud image:
            wordcloud1 = WordCloud().generate(text1)
            wordcloud2 = WordCloud().generate(text2)
            # Display the generated image:
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.subheader("Word frequency in the job description:")
            plt.imshow(wordcloud1, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()

            st.subheader("Word frequency in your Resume:")
            plt.imshow(wordcloud2, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()
		
            
            st.subheader("Got less percentage?, don't worry about it.")

            st.subheader("Contact us at contact.jobshie@gmail.com.")

            st.subheader("We will provide you the full resume building guidance and hold your hand till you got your dream job.")

            st.subheader("Contact us now at contact.jobshie@gmail.com")

            


if __name__ == '__main__':
	main()
