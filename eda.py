import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
import matplotlib.pyplot as plt

working_path = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(working_path, "csv_files")
train_file = os.path.join(csv_file, "SMS_train.csv")

df = pd.read_csv(train_file, encoding_errors="ignore")

def wordcloud(df_message, save_image=False):
    from wordcloud import WordCloud, STOPWORDS
    import string
    
    text = " ".join(df_message.astype(str).tolist())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    
    stopwords = set(STOPWORDS)
    text = " ".join(word for word in text.split() if word not in stopwords and len(word) > 3)
    
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("SMS train dataset wordcloud")
    
    if save_image:
        plt.savefig(os.path.join(working_path, "images", "wordcloud.jpg"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        
def get_label_count(df_label):
    return Counter(df_label)
    
wordcloud(df["Message_body"])
print(get_label_count(df["Label"]))

