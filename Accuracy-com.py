import matplotlib.pyplot as plt

# Setting the x values for the accuracy bar
accuracyLSTM=0
accuracySimpleRNN=0
xValues = ["Simple RNN","LSTM"]
yValues = [accuracySimpleRNN,accuracyLSTM]

# Plotting the accuracy bar
plt.bar(xValues, yValues)

# Setting up x and y labels
plt.xlabel("Model")
plt.ylabel("Accuracy")

# Setting up the title
plt.title('Accuracy for different algorithms')
plt.show()
