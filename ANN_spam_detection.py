import pandas as pd 
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# importing spam data set
spams = pd.read_csv('./data/spams.csv')
spams.head(5)
spams.drop(['Unnamed: 0', 'non_alphabetic_ratio'], axis = 1, inplace = True)

# importing news dataset
news = pd.read_csv('./data/news.csv')
news.head(5)
news['topic'] = 'real'
news.drop(['Unnamed: 0', 'date', 'id', 'entities_spacy'], axis = 1, inplace = True)
news.rename({'article': 'text'}, axis = 1, inplace = True)


############## appending two datasets together
data = pd.concat([spams, news], axis=0, join='outer', ignore_index= True)
data['topic'].value_counts()
data['labels'] = data.topic.map({'real':0, 'general':1, 'marketing':2, 'algorithmic':3})
data['labels'].value_counts()


############# pre-processing function

pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r'#\w+ ?'
combined_pat = r'|'.join((pat_1, pat_2))
www_pat = r'www.[^ ]+'
html_tag = r'<[^>]+>'
negations_ = {"isn't":"is not", "can't":"can not","couldn't":"could not", "hasn't":"has not",
            "hadn't":"had not","won't":"will not",
            "wouldn't":"would not","aren't":"are not",
            "haven't":"have not", "doesn't":"does not","didn't":"did not",
             "don't":"do not","shouldn't":"should not","wasn't":"was not", "weren't":"were not",
            "mightn't":"might not",
            "mustn't":"must not"}

negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')


STOPWORDS = stopwords.words('english')

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
 
def clean_text(text):
    stripped = re.sub(combined_pat, '', text)
    stripped = re.sub(www_pat, '', stripped)
    cleantags = re.sub(html_tag, '', stripped)
    lower_case = cleantags.lower()
    neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
    #letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    tokenized_text = word_tokenize(neg_handled.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS]
    #alpha_numeric = [t for t in cleaned_text if t.isalpha()]
    stemmed = [stemmer.stem(w) for w in cleaned_text]
    lem = [lemma.lemmatize(k, pos = 'v') for k in stemmed]
    cleaned = ' '.join(w for w in lem)
    return cleaned


data['cleaned_text'] = data.text.apply(clean_text)  ########## cleaning the text column


### splitting data into training and test set
x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'], data['labels'], test_size = 0.3, random_state = 10)
y_train.value_counts()
y_test.value_counts()

##################### Creating word embeddings for our ANN model: 1-1) Tokenizing
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)

tokenizer.fit_on_texts(x_train)

x_train_vect = tokenizer.texts_to_sequences(x_train)
x_test_vect = tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1


### plot histogram of lenghts of x_train_vect to decide size of padding
x_train_lengths = []
for i in range(len(x_train_vect)):
    x_train_lengths.append(len(x_train_vect[i]))

plt.style.use('ggplot')
plt.hist(x_train_lengths, bins=10, rwidth=0.85)
plt.xlabel('length of article')
plt.ylabel('count')
plt.yscale('log')
plt.show()      # maxlen = 2000 would be good choice


###### 1-2) encoding labels
from keras.utils import to_categorical
y_train_vect = to_categorical(y_train)
y_test_vect = to_categorical(y_test)

######################### 2) padding 
from keras.preprocessing.sequence import pad_sequences

maxlen = 2000

x_train_vect = pad_sequences(x_train_vect, maxlen=maxlen, padding= 'post')
x_test_vect = pad_sequences(x_test_vect, maxlen=maxlen, padding='post')


################## 3) keras embedding layer and creating model
## 3-1) ANN model
from keras.models import Sequential
from keras import layers
from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
embedding_dim = 50

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

model = Sequential()
model.add(layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation = 'relu'))
model.add(layers.Dense(4, activation = 'softmax'))
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(x_train_vect, y_train_vect,
                    epochs=7,
                    verbose=1,
                    validation_data=(x_test_vect, y_test_vect),
                    batch_size=100, class_weight=class_weight)


plot_history(history)


y_pred = model.predict_classes(x_test_vect)

confusion_matrix(y_test, y_pred)

#### plot confusion matrix
def plot_confusion_matrix(cm,

                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.jpg')
    plt.show()

plot_confusion_matrix(cm = confusion_matrix(y_test, y_pred), target_names = ['real', 'general', 'marketing', 'algorithmic'], normalize=False)
