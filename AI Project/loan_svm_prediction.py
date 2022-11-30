from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import pandas as pd


# Reading in the file
df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

# Previews the first 5 rows in the dataset
df.head()

# Removes the 'Personal Loan' column and isolates it into y
y = df.pop('Personal Loan')

# x is the variable that holds the dataset without the 'Personal Loan'
x = df

# Shows the first 5 rows in both variable datasets
y.head()
x.head()

# Splits the data, testing only a small portion
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)

# Sets up the actual model and fits the data
model = SVC()
model.fit(x_train, y_train)

# Predicts who will take out a personal loan based on the data in x_test
y_pred = model.predict(x_test)

# Calculates and displays the percentage of correctly predicted data
metrics.accuracy_score(y_test, y_pred)

# An array that shows the amount correctly predicted versus the incorrect predictions
confusion_matrix(y_test, y_pred)