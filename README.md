# breast_cancer_prediction
Hello!
This is a breast cancer prediction project using machine learning.
The goal of this project is to build a model that can accurately predict whether a tumor is malignant or benign based on various features.
This is my first machine learning project :)
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which contains information about tumors and their characteristics.
The dataset is available from the UCI Machine Learning Repository.
The dataset contains 569 instances and 30 features, including the target variable (diagnosis).

#### **1_ Import Libraries**

 this code imports the necessary libraries like pandas, numpy, and scikit-learn.
 while pandas is used for data manipulation and analysis, numpy is used for numerical computations, and scikit-learn is used for building machine learning models.

#### **2_ Load Dataset**

 there are two ways to load the dataset.
    the first way is to load the dataset from a CSV file using pandas. the code is:
      # Load data, treat '?' as NaN
        dataset = pd.read_csv("breast-cancer-wisconsin.data", header=None, na_values='?')
      # Separate features and target
        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1].values
    the second way which I used is to use import in python that allows you to load the dataset directly from the UCI Machine Learning Repository.

### **3_ Preprocess Data**

 the code preprocesses the data by fixing problems of rows with missing values.
 I used mean imputation to fill in the missing values.

#### **4- Split Data**

 the code splits the dataset into training and testing sets using train_test_split from scikit-learn.
 the training set is used to train the model, while the testing set is used to evaluate its performance.

#### **5- Train Model**

 I used Logistic Regression as the machine learning model.
 the model is trained using the training set.

#### **6- Evaluate Model**

 the model is evaluated using the testing set. 
 we can also see the confusion matrix here:
    [[82  3]
    [1 54]]
 while 82 is the number of correct predictions that the tumor is benign,
 3 is the number of incorrect  predictions that the tumor is benign,
 1 is the number of incorrect  predictions that the tumor is malignant,
 and 54 is the number of correct predictions that the tumor is malignant.
 the accuracy score is calculated to measure the model's performance.


