# DeepLearningModel_CharityFundingPredictor

# Overview:
Alphabet Soup, a non-profit foundation, seeks to develop an algorithm that can predict the success of funding applicants. Using machine learning and neural networks, our task is to build a binary classifier utilizing the dataset's features. This classifier should determine whether an applicant will be successful if funded by Alphabet Soup.

# Code Overview:
Overall, this code performs data preprocessing tasks like dropping columns, binning low-frequency values, and converting categorical data to numerical format. Then, it builds a deep neural network model, trains it on the preprocessed data, evaluates its performance, and finally saves the trained model to an .h5 file for future use or deployment.

# Libraries:
The necessary libraries for data preprocessing and model building used are:
- pandas, 
- tensorflow,
- train_test_split from `sklearn.model_selection`, and 
- StandardScaler from `sklearn.preprocessing`.

# Loading and Preprocessing Data:
For loading and preprocessing the data, the basic instructions were followed that includes:
   - The code loads data from an external CSV file (charity_data.csv) into a DataFrame.
   - It drops two non-beneficial columns, 'EIN' and 'NAME', from the DataFrame.
   - The number of unique values in each column of `application_df1` is determined using the `nunique()` function.
   - The code identifies and bins low-frequency values in the 'APPLICATION_TYPE' column and replaces them with the label 'Other'.
   - Similarly, it identifies and bins low-frequency values in the 'CLASSIFICATION' column and replaces them with the label 'Other'.
   - The categorical data in the DataFrame is converted to numeric data using one-hot encoding with `pd.get_dummies`.

# Splitting Data: 
The DataFrame `application_df1` is split into features (`X`) and the target variable (`y`). It is further split into training and testing datasets using `train_test_split` from `sklearn.model_selection`.

# Scaling Data: 
The features in `X` are scaled using `StandardScaler` from `sklearn.preprocessing`.
Model Building:
   - A deep neural network model is defined using `tf.keras.models.Sequential`.
   - The model has three layers: two hidden layers and an output layer.
   - The first hidden layer has 80 neurons and uses the ReLU activation function.
   - The second hidden layer has 30 neurons and uses the ReLU activation function.
   - The output layer has one neuron and uses the sigmoid activation function for binary classification.

# Model Compilation and Training:
   - The model is compiled using binary cross-entropy as the loss function, the Adam optimizer, and 'accuracy' as the evaluation metric.
   - The model is trained on the training data (`X_train_scaled` and `y_train`) for 50 epochs using `nn.fit()`.

# Model Evaluation:
The trained model is evaluated using the test data (`X_test_scaled` and `y_test`) with the `evaluate()` method, and the loss and accuracy metrics are printed.
![SequentialModel]()

![Loss&Accuracy]()
# Model Export: 
The model is saved as an .h5 file named "AlphabetSoupCharity.h5" using `nn.save()`. The `files.download()` function is then used to download the saved model file to the local machine.

# Optimize the Model:
Since the loss was too high so, few attempts were made to optimize the model trying to achieve atleast 75% accuracy and less loss. 

## Optimization 1:
![Optimization Tweak1]()
![Sequential Model1]()
![Loss&Accuracy1]()

## Optimization 2:
![Optimization Tweak2]()
![Sequential Model2]()
![Loss&Accuracy2]()

## Optimization 3:
![Optimization Tweak3]()
![Sequential Model3]()
![Loss&Accuracy3]()


# Conclusion:
Overall, the loss was very high and the accuracy for all three optimizations did not make it to 75%. The best results for accuracy were achieved during the original attempt, however, the loss was 0.57 the highest. I tried several different ways using different activation functions, optimization, tweaking dataset, and few other ways but there was no such change observed. 
All three attempts to improve the accuracy of the model did not succeed; instead, they led to an increase in or higher loss. One of the ways to improve the result would have been to try to use Keras Tuner HyperModel. This would allow us to predict the best fit. 
Since our dataset contains a lot of categorical data, it might be worth considering trying decision tree or random forest methods, as they could potentially perform better in this scenario.




