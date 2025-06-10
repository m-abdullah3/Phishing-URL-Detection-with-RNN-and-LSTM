import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd #to read and manage the dataset
import tensorflow as tf # for deep learning models
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
# pad_sequences is used to pad a sequence with zeros if the sequence length is less than the required length
from tensorflow.keras.preprocessing.text import Tokenizer # used for text partitioning
from tensorflow.keras.models import Sequential # we will use the sequence model for sequence data analysis
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D # layers to be used to build the DL model
from tensorflow.keras.layers import Input, Embedding
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the dataset as a pandas frame
dataFrame=pd.read_csv("phishing_site_urls.csv")

#shape of the dataset
print ("shape = ",dataFrame.shape)

#Printing 1st five rows using head()
print("Head: ")
print(dataFrame.head())
print("------------------------------------------------")

#Printing last five rows using tail()
print("Tail")
print(dataFrame.tail())
print("------------------------------------------------")

#Finding Missing Values
print("Missing Values")
print(dataFrame.isnull().sum())
print("------------------------------------------------")



#Finding Duplicate values
print("number of Duplicates = ",dataFrame.duplicated().sum())

#Removing duplicates
dataFrame.drop_duplicates(inplace=True)
print("Removing Duplictes..........")
print("number of Duplicates = ", dataFrame.duplicated().sum())

print("------------------------------------------------")

#Checking count of unique labels
print("Count of Labels")
print(dataFrame["Label"].value_counts())
print("------------------------------------------------")

#Getting information about the dataframe using info()
print("Info()")
print(dataFrame.info())
print("------------------------------------------------")

#Encoding non-numeric values to numeric
labelEncoder=preprocessing.LabelEncoder()
dataFrame["Label"]=labelEncoder.fit_transform(dataFrame["Label"])

print("Encoded Labels")
print(dataFrame["Label"].value_counts())
print("------------------------------------------------")

tokenizer=Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(dataFrame['URL'])
url_sequences = tokenizer.texts_to_sequences(dataFrame['URL'],)

max_length = 100
padded_sequences = pad_sequences(url_sequences,padding="post",maxlen=max_length)

print("Tokenizarion in process......................... done")
print("Orignal URL")
print(dataFrame["URL"][0])
print("------------------------------------------------")
print("Sequence generated from the URL")
print(url_sequences[0])
print("------------------------------------------------")
print("The padded Sequence")
print(padded_sequences[0])
print("------------------------------------------------")


# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, dataFrame['Label'], test_size=0.20, random_state=42
)

# Model Building
model = Sequential()
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
print("Vocabulary Size = ",vocab_size)
print("-----------------------------------------------------------------------------")

embeddingVectoreLen = 200  # Dimension of word embeddings
model.add(Input(shape=(100,)))  # Define input shape here
model.add(Embedding(vocab_size, embeddingVectoreLen))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','recall',"precision"])

# Summarize the model
model.summary()

TrainModel= model.fit(
    X_train,
    y_train,
    epochs=1,
    batch_size=32,
    validation_split=0.2,
    verbose=1,

)

# Evaluate the model
loss, accuracy, recall, precision= model.evaluate(X_test, y_test, verbose=0)
f1_score= 2 * (precision * recall) / (precision + recall)

y_pred=model.predict(X_test)

modelName="LSTM"

# Calculating the Confusion Matrix
confusionMatrix =tf.math.confusion_matrix(y_test,y_pred)


# Plotting the confusion matrix using Seaborn's heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix for {modelName}')
plt.show()

# Storing the performance metrics in a dictionary
results = [{
    'Model': modelName,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score,
    'True Negative': confusionMatrix[0][0].numpy(),
    'True Positive': confusionMatrix[1][1].numpy(),
    'False Positive': confusionMatrix[0][1].numpy(),
    'False Negative': confusionMatrix[1][0].numpy(),
}]

# Converting the results list into a DataFrame
resultsDataFrame = pd.DataFrame(results)

print("-----------------------------------------------------------------------------")
# Printing the DataFrame containing the results by converting into a string
print(resultsDataFrame.to_string(index=True, max_rows=None, max_cols=None, line_width=None, float_format='{:,.4f}'.format, header=True))

# Setting the x values for the accuracy bar
xValues = [modelName]
yValues = [accuracy]

# Plotting the accuracy bar
plt.bar(xValues, yValues)

# Setting up x and y labels
plt.xlabel("Model")
plt.ylabel("Accuracy")

# Setting up the title
plt.title(f'Accuracy for = {modelName}')
plt.show()




