# TASK--6

## Objective :
  To understand and implement the KNN algorithm for solving classification problems

## Used :
 - Scikit-learn
 - Pandas
 - Matplotlib

## Work Flow :
 1. Data Preprocessing
  - Loaded the dataset
  - Dropped the Id column
  - Separated features and target
  - Normalized features using StandardScaler

 2. Model Building
  - Used KNeighborsClassifier from sklearn
  - Tried multiple values of K (1 to 10)
  - Selected the best K = 5 based on accuracy
    
 3. Model Evaluation
  - Evaluated the model using:
  - Accuracy score
  - Confusion matrix

 4. Visualization
  - Plotted the confusion matrix
  - Visualized decision boundaries (using first 2 features)

## Results :
 - Best Accuracy: Achieved at K=5

Insights:
  - KNN is a simple and effective classification method
  - Performance depends on proper feature scaling and K value


## Output :

K = 1, Accuracy = 0.98
K = 2, Accuracy = 0.98
K = 3, Accuracy = 1.00
K = 4, Accuracy = 0.98
K = 5, Accuracy = 1.00
K = 6, Accuracy = 1.00
K = 7, Accuracy = 1.00
K = 8, Accuracy = 1.00
K = 9, Accuracy = 1.00
K = 10, Accuracy = 1.00

Confusion Matrix:

![pic](https://github.com/user-attachments/assets/596e465c-5417-4ec3-ace2-f23b4c296e1c)

![picc](https://github.com/user-attachments/assets/19ee32d5-163b-4414-9f4f-c9f395912ae3)
