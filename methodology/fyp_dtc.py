# data manipulation
import pandas as pd

# for data visualisation
import matplotlib.pyplot as plt
import seaborn as sb

# for non-numeric to numeric convertion
from sklearn.preprocessing import LabelEncoder

# decision tree model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree

# model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def dtc(df, filename):
    '''
    classify decision column using decision tree classifier, and return evaluation scores, confusion matrix, and decision tree
    
    Parameters: 
        df (Pandas Dataframe): dataframe with data
        filename (String): unique name for decision tree file
    '''
    
    # labelencoder to change non-numerical fields to numerical
    encoder = LabelEncoder()

    # fit_transform encoder into each non-numerical column
    df['Decision'] = encoder.fit_transform(df['Decision'])
    
    # print unique values of decision field
    # 0: buy; 1: neutral; 2: sell
    print(df['Decision'].value_counts(ascending = True))
    
    # features and target of the dataframe
    features = df.drop(['Decision', 'Date'], axis = 1)
    target = df['Decision']
    
    # map features and target into X and y variables
    X = features.iloc[:,:]
    y = target.iloc[:,]

    # splitting X and y variables into train and test sets, 80% train set - 20% test set
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 33)
    
    # defining decision tree classifier
    dtc = DecisionTreeClassifier()

    # fitting train sets into decision tree model
    dtc = dtc.fit(train_x, train_y)

    # predict test data using fitted decision tree classifier model
    dtc_pred = dtc.predict(test_x)
    
    # evaluating decision tree classifier model
    # accuracy score
    print(f'Accuracy Score Percentage: {round((accuracy_score(test_y, dtc_pred)*100), 2)}%')
    # precision score
    print(f'Precision Score Percentage: {round((precision_score(test_y, dtc_pred, average="macro", zero_division = 0)*100), 2)}%')
    # recall score
    print(f'Recall Score Percentage: {round((recall_score(test_y, dtc_pred, average="macro")*100), 2)}%')
    # f1 score
    print(f'F1 Score Percentage: {round((f1_score(test_y, dtc_pred, average="macro",zero_division = 0)*100), 2)}%')
    
    # confusion matrix
    cm = confusion_matrix(test_y, dtc_pred)

    # plotting confusion matrix
    fig = ConfusionMatrixDisplay(cm)
    fig.plot()

    plt.title('Decision Tree Classifier Confusion Matrix')

    plt.show()
    
    # decision tree visualisation
    plt.figure(figsize = (18, 9))
    plot_tree(dtc)
    plt.savefig(f'decision_tree_{filename}.png')
    
    