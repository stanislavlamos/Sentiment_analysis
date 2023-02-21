import fasttext
import sklearn
import csv
import re


def preprocess_sentiment140_dataset():
    train = open('./data/tweets.train','w')
    test = open('./data/tweets.valid','w')
    with open('./data/training.1600000.processed.noemoticon.csv', mode='r', encoding = "ISO-8859-1") as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=['target', 'id', 'date', 'flag', 'user', 'text'])
        line = 0
        for row in csv_reader:
            # Clean the training data
            # First we lower case the text
            text = row["text"].lower()
            # remove links
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
            #Remove usernames
            text = re.sub('@[^\s]+','', text)
            # replace hashtags by just words
            text = re.sub(r'#([^\s]+)', r'\1', text)
            #correct all multiple white spaces to a single white space
            text = re.sub('[\s]+', ' ', text)
            # Additional clean up : removing words less than 3 chars, and remove space at the beginning and teh end
            text = re.sub(r'\W*\b\w{1,3}\b', '', text)
            text = text.strip()
            line = line + 1
            # Split data into train and validation
            if line%16 == 0:
                print(f'__label__{row["target"]} {text}', file=test)
            else:
                print(f'__label__{row["target"]} {text}', file=train)

def train_fasttext():
    model_tweet = fasttext.train_supervised('./data/tweets.train')
    print(model_tweet.test('./data/tweets.valid'))
    model_tweet.save_model('model.bin')


if __name__ == '__main__':
   #preprocess_sentiment140_dataset()
   train_fasttext()