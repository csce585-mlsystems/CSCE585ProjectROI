# Purpose: This file will contain code that helps with Model Development.
# Body of neccessary imports
import os
import sys
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pdb as pb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist

DemoMode = True
sys.path.insert(3,"C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo" if DemoMode == False else "ProjectRepo")
# end of body of neccessary imports
# Body of neccessray imports for new verison of model

from sklearn.datasets import fetch_openml #<-- USed to fetch MNIST Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier #<-- NOTE: Can use this in place of model??
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# End of Body of neccessray imports for new verison of model
# Body of modes of plots
# baseLinePerform, Experiment1, Experiment2, Experiment3 = True,False,False,False #<-- Mode #1
baseLinePerform, Experiment1, Experiment2, Experiment3 = False,True,False,False #<-- Mode #2
# baseLinePerform, Experiment1, Experiment2, Experiment3 = False,False,True,False #<-- Mode #3
# baseLinePerform, Experiment1, Experiment2, Experiment3 = False,False,False,True #<-- Mode #4
# End of Body of modes of plots
# Body of setting up paths for saving models"
current_dir = os.path.dirname(__file__).replace("\\","/")
sys.path = [sys.path[i].replace("\\","/") for i in range(len(sys.path))]
# End of Body of setting up paths for saving models
# NOTE: For function, will need to have an array of bools that reference the modes that dictate which plot number to assign to prints.
# Important Steps in Model Development: 1) Obtaining the training data, 2) Create the model containing initialized weights and a bias[which would be very involved with using numbers from certain columns], 3) Observe model's performance before training, 4) Defining a loss function for model, 5) Write a basic Training Loop

def attempt3():
    inNoteBook = False
    # Body of Neccessary Imports for Model Development.
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.utils import to_categorical, plot_model
    from tensorflow.keras.datasets import mnist
    # End of Body of Neccessary Imports for Model Development.
    # NOTE: Below is used to modify network parameters using experiment setup func(s).
    global x_train, y_train, x_test, y_test
    global x_train; global x_test; 
    demo = True
    parentDir = "C:/Users/adoct/Notes for CSCE Classes\\[Fall 2025\\]/Notes for CSCE 585/projectCode" if demo == False else os.getcwd().replace("\\","/")
    filePathToModelDir = f"{parentDir}/projectCode/MLLifecycle/ModelDevelopmentAndTraining/preparedDataset.csv" if inNoteBook == False else "preparedDataset.csv"
    train_stocks = pd.read_csv(f"{filePathToModelDir}"); train_labels = test_labels = train_stocks.loc[:,"Company"] #<-- References labels which are derived from custom engineered dataset.
    del train_stocks["Unnamed: 0"]
    test_stocks = pd.read_csv(f"{filePathToModelDir}"); test_labels = train_stocks.loc[:,"Company"] #<-- References labels which are derived from custom engineered dataset.
    del test_stocks["Unnamed: 0"]
    # x_train = train_stocks.loc[:,train_stocks.columns != "Optimality"]
    x_train = train_stocks.loc[:,:]
    y_train = train_labels
    # x_test = test_stocks.loc[:,test_stocks.columns != "Optimality"]
    x_test = test_stocks.loc[:,:]
    y_test = test_labels
    # ^^ Above ensures that prediction labels are y and x references the data used to make said decision.
    # "Compute the number of labels"
    num_labels = len(np.unique(y_train))
    # "Convert to one-hot vector"[we converted the labels to one-hot vectors using to_categorical]
    # y_train = to_categorical(y_train.index)
    # y_test = to_categorical(y_test.index)
    input_size = len(train_stocks.columns)  #<-- input_size refers to number of attributes for each row tuple of data.
    # Establishing Network Parameters
    global batch_size, hidden_units, dropout
    global model #<-- Have this here, so Experimental Setup functions can tweak model as needed.
    batch_size = 128
    hidden_units = 256
    dropout = 0.45
    isExperimentSetup3Active = False
    def newModelIntegration(modelNew = False):
        if(not modelNew):
            model = Sequential()
        else:
            isExperimentSetup3_newModelActive = False
            global x_train, x_test
            global y_train, y_test
            # NOTE: Insert body of newModel here!
            def experimentSetup0_newModel():
                Tutorial = False
                if(Tutorial):
                    # This will reference the default settings irrespective to experiments.
                    # 1) Creating the Model.
                    print("---Undergoing Experiment Setup #0_newModel---")
                    print("---BEGINNING OF PART 1) of Code Example")
                    pb.set_trace()
                    ## 1) Body of Fetching MNIST Dataset
                    mnist = fetch_openml('mnist_784', version=1)
                    mnist.keys()
                    print("---END OF PART 1) of Code Example")
                    pb.set_trace()
                    ## Part 2) Body of Fetching MNIST Dataset
                    X, y = mnist["data"], mnist["target"]
                    X.shape
                    y.shape

                    import matplotlib as mpl
                    import matplotlib.pyplot as plt

                    # some_digit = X[0] <-- colnd't do this way 
                    some_digit = X.loc[0,:].to_numpy() # <-- colnd't do this way 
                    some_digit_image = some_digit.reshape(28,28) 
                    plt.imshow(some_digit_image,cmap=mpl.cm.binary, interpolation='nearest')
                    plt.axis("off")
                    plt.show()
                    
                    y[0]

                    y = y.astype(np.uint8)

                    X_train, X_test, y_train, y_test = X[:int(6e4)],X[int(6e4):],y[:int(6e4)],y[int(6e4):]

                    
                    y_train_5 = (y_train == 5) # <-- "True for all 5s, False for all other digits"
                    y_test_5 = (y_test == 5) 

                    from sklearn.linear_model import SGDClassifier #<-- NOTE: Can use this in place of model??

                    sgd_clf = SGDClassifier(random_state=42)
                    sgd_clf.fit(X_train, y_train_5) #<-- This command is used to train the model

                    sgd_clf.predict([some_digit]); #<-- This command is used to TEST/evalutate the model. 

                    from sklearn.model_selection import cross_val_score
                    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form

                    from sklearn.base import BaseEstimator

                    class Never5Classifier(BaseEstimator):
                        def fit(self, X, y=None):
                            pass
                        def predict(self, X):
                            return np.zeros((len(X), 1), dtype=bool)
                    
                    never_5_clf = Never5Classifier()
                    cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

                    from sklearn.model_selection import cross_val_predict
                    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

                    from sklearn.metrics import confusion_matrix
                    confusion_matrix(y_train_5, y_train_pred)


                    from sklearn.metrics import precision_score, recall_score
                    precision_score(y_train_5, y_train_pred) # == 4096/(4096*1522) = \text{precision} = \frac{TP}{TP + FP}
                    recall_score(y_train_5, y_train_pred) # == 4096/(4096*1522) = \text{recall} = \frac{TP}{TP + FN}

                    from sklearn.metrics import f1_score
                    f1_score(y_train_5, y_train_pred) 

                    y_scores = sgd_clf.decision_function([some_digit])
                    y_scores

                    threshold = 0
                    y_some_digit_pred = (y_scores > threshold)

                    # Increasing Threshold
                    threshold = 8000
                    y_some_digit_pred = (y_scores > threshold)


                    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

                    from sklearn.metrics import precision_recall_curve

                    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores);

                    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
                        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
                        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

                        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                        plt.show()


                    threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)] # == 7813 


                    precision_score(y_train_5, y_train_pred)
                    recall_score(y_train_5, y_train_pred)

                    

                    print("---END OF PART 2) of Code Example")
                    print("---BEGINNING OF PART 3) of Code Example")
                    pb.set_trace()
                    from sklearn.metrics import roc_curve

                    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

                    def plot_roc_curve(fpr, tpr, label=None):
                        plt.plot(fpr,tpr,linewidth=2,label=label)
                        plt.plot([0,1], [0,1], 'k--') # "Dashed diagonal"
                                # [...] #<-- "Add axis labels and grid"
                        plot_roc_curve(fpr,tpr)
                        plt.show()

                    from sklearn.metrics import roc_auc_score

                    roc_auc_score(y_train_5, y_scores)
                    print("---This is the point where I believe all is needed for model development stuff---")
                    pb.set_trace()

                    from sklearn.ensemble import RandomForestClassifier

                    forest_clf = RandomForestClassifier(random_state=42)
                    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

                    y_scores_forest = y_probas_forest[:,1] # "Score = proba of positive class"
                    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

                    plt.plot(fpr, tpr, "b:", label="SGD")
                    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
                    plt.legend(loc="lower right")
                    plt.show()

                    roc_auc_score(y_train_5, y_scores_forest)

                    sgd_clf.fit(X_train,y_train) # "y_train, not y_train_5"
                    sgd_clf.predict([some_digit]) # "y_train, not y_train_5"

                    
                    some_digits_scores = sgd_clf.decision_function([some_digit])

                    np.argmax(some_digits_scores)

                    sgd_clf.classes_
                    sgd_clf.classes_[5]
                    sgd_clf.classes_[np.argmax(some_digits_scores)]

                    from sklearn.multiclass import OneVsOneClassifier
                    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
                    ovo_clf.fit(X_train,y_train)
                    ovo_clf.predict([some_digit])
                    len(ovo_clf.estimators)

                    forest_clf.fit(X_train,y_train)
                    forest_clf.predict([some_digit])

                    forest_clf.predict_proba([some_digit])
                    
                    cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring="accuracy")

                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
                    cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

                    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
                    conf_mx = confusion_matrix(y_train,y_train_pred)
                    conf_mx

                    plt.matshow(conf_mx, cmap=plt.cm.gray)
                    plt.show()

                    row_sums = conf_mx.sum(axis=1, keepdims=True)
                    norm_conf_mx = conf_mx/row_sums

                    np.fill_diagonal(norm_conf_mx, 0)
                    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
                    plt.show()
                    ## End of 1)
                    print("---END OF PART 3) of Code Example")
                    pb.set_trace()
                    ## Part 2) End of Body of Fetching MNIST Dataset
                    print("---BEGINNING OF PART 4) of Code Example")
                    pb.set_trace()
                    ## Part 3) Body of Fetching MNIST Dataset
                else:
                    # Body of loading in data for classifiers to consume to make predictions
                    print("---Loading in data for classifier consumption---")
                    # global x_train, y_train, x_test, y_test
                    # x_train = train_stocks.loc[:,train_stocks.columns != "Optimality"]
                    # y_train = train_labels
                    # x_test = test_stocks.loc[:,train_stocks.columns != "Optimality"]
                    # y_test = test_labels
                    # ^^ Above ensures that prediction labels are y and x references the data used to make said decision.
                    # "Compute the number of labels"
                    # num_labels = len(np.unique(y_train))
                    # "Convert to one-hot vector"[we converted the labels to one-hot vectors using to_categorical]
                    # y_train = to_categorical(y_train.index)
                    # y_test = to_categorical(y_test.index)
                    # input_size = len(train_stocks.columns)  #<-- input_size refers to number of attributes for each row tuple of data.

                    # ^^ Using above to have table to reference variables to be utilized. 
                    

                    print("---End of Loading in data for classifier consumption---")
                    # End of Body of loading in data for classifiers to consume to make predictions
                    # Body of creating classifiers to be used by model
                    print("---Creating Classifiers to for model choices---")
                    from sklearn.linear_model import SGDClassifier #<-- NOTE: Can use this in place of model??
                    
                    sgd_clf = SGDClassifier(random_state=42)
                    # NOTE: Between this, need to have: i) Model needs to be trained on optimality, ii) Model needs to be in able to apply optimality number to each company given, iii) company labels need to be one-hot encoded or transformed into integers for model intepretation. 

                    pb.set_trace()
                    x_train["Company"] = pd.Series(x_train["Company"].index.tolist())                    
                    """
                    NOTE: Below isn't needed, remove in final version.
                    # y_train = pd.Categorical(y_train.index.tolist())
                    # y_test = pd.Categorical(y_test.index.tolist())
                    # sgd_clf.fit(X_train, y_train) #<-- This command is used to train the model
                    # sgd_clf.fit(x_train, y_train) #<-- This command is used to train the model
                    """
                    if(x_train.isna().any(axis = 1).sum()):
                        for i in x_train.columns:
                            x_train[i].fillna(x_train[i].mean())
                    
                    x_test["Company"] = pd.Series([i for i in range(len(x_test["Company"]))])
                    sgd_clf.fit(x_train, x_train["Company"]) #<-- This command is used to train the model
                    
                    
                    # sgd_clf.predict(y_test); #<-- This command is used to TEST/evalutate the model. 
                    sgd_clf.predict(x_test); #<-- NOTE: x_test is used here b/c scikit-learn automatically omitts the label column.
                    pb.set_trace() 
                    
                    
                    
                    
                    
                    
                    
                    print("---End of Creating Classifiers to for model choices---")
                    # End of Body of creating classifiers to be used by model


                    # Body of evaluating classifier's results via plotting etc
                    print("---Evaluating Classifier's Results---")
                    
                    from sklearn.model_selection import cross_val_score
                    # cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                    y_train = x_train["Company"]
                    cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                    from sklearn.model_selection import cross_val_score
                    cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                    cross_val_score(never_5_clf, x_train, y_train, cv=3, scoring="accuracy")
                    from sklearn.model_selection import cross_val_predict
                    y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)
                    from sklearn.metrics import confusion_matrix
                    confusion_matrix(y_train, y_train_pred)
                    y_scores = cross_val_predict(sgd_clf, x_train, y_train, cv=3, method="decision_function")
                    from sklearn.metrics import precision_recall_curve

                    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores);

                    print("---End of Evaluating Classifier's Results---")

                    
                    
                    # End of Body of evaluating classifier's results via plotting etc

                    # Body of tweaking the precison/recall tradeoff to make the perfect classifier model
                    print("---Tweaking precision/recall tradeoff to make perfect classifier---")
                    
                    
                    print("---End of Tweaking precision/recall tradeoff to make perfect classifier---")

                    
                    
                    # End of Body of tweaking the precison/recall tradeoff to make the perfect classifier model
                print("---End of Experiment Setup #0_newModel---")
                return
            def experimentSetup1_newModel():
                    # Goal of exp: Want to see how model params affect the acuaracy of the model by modifying batch size and hidden units and dropout[UPDATE: Instead of these three params, will use threshold, precision, and recall instead!]
                    print("---Undergoing Experiment Setup #1_newModel---")
                    pb.set_trace()
                    setN = 1 #<-- Change number for this to get values from resp sets.
                    experiment1Tuples: list[tuple] = [(5,45), (5,30), (5,3), ("""NOTE: Other tuples can change one or more parameters whilst keeping at least one constant""")]
                    #batch_size = experiment1Tuples[setN][0]
                    #hidden_units = experiment1Tuples[setN][1]
                    #dropout = experiment1Tuples[setN][2]
                    
                    # NOTE: Need to replace metrics above with classifier parameters. Need to do more invesigating. 
                    print("---End of Experiment Setup #1_newModel---")
                    return
            global listVerOfX_train
            listVerOfX_train = x_train.columns[:len(x_train.columns)-1].to_list()
                        # ^^ Utilized to setup experiment #2 whose desc is below.
            def experimentSetup2_newModel(numQuantLvls = 2): # NOTE: experimentSetup2_newModel can stay the same. 
                # Goal of exp: Want to see model performance based on degree of quantanization of data
                print("---Undergoing Experiment Setup #2_newModel---")
                pb.set_trace()
                # NOTE: Below will involve replacing 255 with a different number based on degree of quantanization of data.
                # Using for loop to iterate through each column to apply this quantinization to each column.
                global x_train, x_test
                for i in listVerOfX_train:
                    x_trainMax = x_train[i].max()
                    x_trainMin = x_train[i].min()
                    x_testMax = x_test[i].max()
                    x_testMin = x_test[i].min()
                    widthsOfQuant = [(x_trainMax - x_trainMin)/(numQuantLvls - 1),(x_testMax - x_testMin)/(numQuantLvls - 1)]
                    x_train.loc[:,i] = x_train.loc[:,i]/widthsOfQuant[0]
                    x_test.loc[:,i] = x_test.loc[:,i]/widthsOfQuant[0]
                print("---End of Experiment Setup #2_newModel---")
                return
            def experimentSetup3_newModel(pair1 = True):
                # Goal of exp: Will be responsible for removing feature-pairs[complete]
                print("---Undergoing Experiment Setup #3_newModel---" if isExperimentSetup3_newModelActive else "---Experiment Setup #3_newModel was skipped---")
                p
                if(pair1):
                    del x_test[['P/E', 'P/B']]
                    del x_train[['P/E', 'P/B']]
                else:
                    del x_test[['NCAV', 'Share Price']]
                    del x_train[['NCAV', 'Share Price']]
                print("---End of Experiment Setup #3_newModel---" if isExperimentSetup3_newModelActive else "---End of Experiment Setup #3_newModel was skipped---")
                return model.add(Activation(activationFuncs[0])) if isExperimentSetup3_newModelActive == True else None
            def experimentSetup4_newModel():
                print("---Undergoing Experiment Setup #4_newModel---")
                pb.set_trace()
                # Need to have an experiment that utilizes scaling analysis[in progress]
                print("---End of Experiment Setup #4_newModel---")
                return
            
            experimentSetup0_newModel()
            # experimentSetup1_newModel()
            experimentSetup2_newModel()
             
    # newModelIntegration(False)
    pb.set_trace()
    newModelIntegration(True)
    def experimentSetup0():
       # This will reference the default settings irrespective to experiments.
       # 1) Creating the Model.
        print("---Undergoing Experiment Setup #0---")
        model.add(Input((x_train.shape[1],)))
        model.add(Dense(hidden_units))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units))
        model.add(Activation('relu')) if experimentSetup3() == None else experimentSetup3() #<-- This changes the Activation function, adhering to experimentSetup3!
        model.add(Dropout(dropout))
        model.add(Dense(num_labels))
        print("---End of Experiment Setup #0---")
        return
    def experimentSetup1():
        # Goal of exp: Want to see how model params affect the acuaracy of the model by modifying batch size and hidden units and dropout
        print("---Undergoing Experiment Setup #1---")
        setN = 1 #<-- Change number for this to get values from resp sets.
        experiment1Tuples: list[tuple] = [(128,256,0.45), (64,128,0.45), (128,256,0.3), ("""NOTE: Other tuples can change one or more parameters whilst keeping at least one constant""")]
        batch_size = experiment1Tuples[setN][0]
        hidden_units = experiment1Tuples[setN][1]
        dropout = experiment1Tuples[setN][2]
        print("---End of Experiment Setup #1---")
        return
    global listVerOfX_train
    listVerOfX_train = x_train.columns[:len(x_train.columns)-1].to_list()
    # ^^ Utilized to setup experiment #2 whose desc is below.
    def experimentSetup2(numQuantLvls = 2):
        # Goal of exp: Want to see model performance based on degree of quantanization of data
        print("---Undergoing Experiment Setup #2---")
        # NOTE: Below will involve replacing 255 with a different number based on degree of quantanization of data.
        # Using for loop to iterate through each column to apply this quantinization to each column.
        global x_train, x_test
        for i in listVerOfX_train:
            x_trainMax = x_train[i].max()
            x_trainMin = x_train[i].min()
            x_testMax = x_test[i].max()
            x_testMin = x_test[i].min()
            widthsOfQuant = [(x_trainMax - x_trainMin)/(numQuantLvls - 1),(x_testMax - x_testMin)/(numQuantLvls - 1)]
            x_train.loc[:,i] = x_train.loc[:,i]/widthsOfQuant[0]
            x_test.loc[:,i] = x_test.loc[:,i]/widthsOfQuant[0]
        print("---End of Experiment Setup #2---")
        return
    def experimentSetup3():
        # Goal of exp: Want to see model performance based on type of activation function from a subset of all possible activation functions.
        print("---Undergoing Experiment Setup #3---" if isExperimentSetup3Active else "---Experiment Setup #3 was skipped---")
        activationFuncs = ['elu', 'sigmoid', 'tanh' ]
        print("---End of Experiment Setup #3---" if isExperimentSetup3Active else "---End of Experiment Setup #3 was skipped---")
        return model.add(Activation(activationFuncs[0])) if isExperimentSetup3Active == True else None
    def experimentSetup4():
        print("---Undergoing Experiment Setup #4---")
        # Need to have an experiment that utilizes scaling analysis[in progress]
        print("---End of Experiment Setup #4---")
        return
    experimentSetup0()
    # experimentSetup1()
    experimentSetup2()
    # This is the output for one-hot vector, being sent into the softmax activation function.
    model.add(Activation('softmax'))
    model.summary()
    # 4) Defining the loss function and compiling model
    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
    # 5) Training the network
    x_train.loc[:,"Company"] = pd.Series([2**i for i in range(x_train.shape[0])])
    x_test.loc[:,"Company"] = pd.Series([2**i for i in range(x_train.shape[0])])
    x_train = x_train.astype("float64")
    x_test = x_test.astype("float64")
    x_train.loc[:,"Company"] = x_train.loc[:,"Company"].astype("int32")
    x_test.loc[:,"Company"] = x_test.loc[:,"Company"].astype("int32")
    x_trainCopy = tf.data.Dataset.from_tensor_slices((x_train.values.astype(np.float32),tf.convert_to_tensor(y_train).numpy().astype(np.float32)))
    x_testCopy = tf.data.Dataset.from_tensor_slices((x_test.values.astype(np.float32),tf.convert_to_tensor(y_test).numpy().astype(np.float32)))
    train_history = model.fit(x_train,y_train,epochs=20, batch_size=batch_size)
    # 6) Evaluating Model
    acc = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc[1]))
    
    # Body of making copy of test_stocks to be used later
    test_stocksCopy = test_stocks

    # End of Body of making copy of test_stocks to be used later

    test_stocks["Company"] = test_stocks.index; test_stocks.loc[:,"Company"] = test_stocks.loc[:,"Company"].astype("int32")
    # 6) Verifying and Visualzing the Predictions
    # a) Obtaining accuarrcy of the predictions:
    WriteModelToAFile = True
    predictions = model.predict(x_test) #<-- NOTE: This retunrns an array of probabilities for each class. Thus, for future consumption by person, could assign these predictions to a column added to test_stocks?
    # Write code for writing model to a file here
    local = False #<-- '""' Needs to refer to virtual environment. [VIRTUAL ENVIRONMENT ADDRESS THING[NOTE]: Will need to change this file path to adhere to virtual environment!] 
    ImprovedAlgo = True
    if(WriteModelToAFile):
        # Goal: Make copy of test stocks, add predictions column referencing model's 
        print("---Writing Model to a file for future use---")
        
        copyTwoSendOff = test_stocks
        # copyTwoSendOff["Model Predictions"] = predictions
        # copyTwoSendOff.loc[:,"Model Predictions"] = predictions
        # copyTwoSendOff.loc[:,"Model Predictions"] = pd.Series([np.argmax(x) for x in predictions])
        # Body of modifying predictions to ensure that each prediction results in only ONE thing being chosen
        # Attempt #1[closest I've gotten]
        list_a = test_stocks["Optimality"].tolist()
        list_b = [np.argsort(predictions[i])[-1] for i in range(len(predictions)) if i < len(predictions) - 1 and np.argsort(predictions[i][-1]) in np.argsort(predictions[i+1][-1])]
        list_b = [x.item() for x in list_b]
        missing_element = set(list_a).difference(set(list_b)) #<-- equal to list_a \cap (list_b)^c . 
        copyTwoSendOff["Model Predictions"] = pd.Series([np.argmax(x) for x in predictions]) if ImprovedAlgo == False else pd.Series([*list_b,missing_element.pop()]) 
        # thoughts. 
        # End of Attempt #1
        # End of Body of modifying predictions to ensure that each prediction results in only ONE thing being chosen
        copyTwoSendOff.loc[:, "Company"] = test_labels

        
        demo = True
        parentDir = "C:/Users/adoct/Notes for CSCE Classes\\[Fall 2025\\]/Notes for CSCE 585/projectCode" if demo == False else os.getcwd().replace("\\","/")
        writePathForModelPreds = f"{parentDir}/projectCode/MLLifecycle/ModelDevelopmentAndTraining/ModelPredictions.csv" if local == True else "ModelPredictions.csv" #<-- '""' Needs to refer to virtual environment. 

        
        
        
        copyTwoSendOff.to_csv(f"{writePathForModelPreds}")
        print("---End of Writing Model to a file for future use---")

        
        
        
        # End of writing code for writing model to a file here
    predictions[0]

    """
    # NOTE: Below will reference ways to transform predictions into string format to be able to utilzied for user consumption[DISREGARD #1: Only applicable to model integration into app]
    print(test_stocks[test_stocks["Optimality"] == np.argmax([predictions[0]])]["Company"].values[0])
    print("----")
    print(test_stocks[test_stocks["Optimality"] == np.argmax([x for x in predictions[1] if predictions[1].tolist().index(x) != np.argmax([predictions[0]])])]["Company"].values[0])
    print("----")
    print(test_stocks[test_stocks["Optimality"] == np.argmax([x for x in predictions[2] if predictions[2].tolist().index(x) != np.argmax([predictions[0]]) and predictions[2].tolist().index(x) != np.argmax([predictions[1]])])]["Company"].values[0])
    print("----")
    print(test_stocks[test_stocks["Optimality"] == np.argmax([x for x in predictions[3] if predictions[3].tolist().index(x) != np.argmax([predictions[0]]) and predictions[3].tolist().index(x) != np.argmax([predictions[1]]) and predictions[3].tolist().index(x) != np.argmax([predictions[2]])])]["Company"].values[0])
    print("----")
    # UPDATE: Above works, BUT the problem is that each prediction MUST be unique, and the predictions should be limited based on the value(s) of the previous prediction.
    # Attempt #1 At solution:
    print(test_stocks[test_stocks["Optimality"] == np.argmax(predictions[0])]["Company"].values[0])
    print(test_stocks[test_stocks["Optimality"] == np.argmax(predictions[1])]["Company"].values[0])
    print(test_stocks[test_stocks["Optimality"] == np.argmax(predictions[2])]["Company"].values[0])
    print(test_stocks[test_stocks["Optimality"] == np.argmax(predictions[3])]["Company"].values[0])
    # End of Attempt #1 At solution:
    # Attempt #2 At solution:
    np.argsort(predictions[0])
    np.argmax(predictions[0])
    [x for x in np.argsort(predictions[1]) if x != np.argmax(predictions[0])]
    np.argsort(predictions[1][np.argsort(predictions[0])[0]])
    [x for x in np.argsort(predictions[1]) if x != np.argmax(predictions[0])]
    np.argsort(predictions[2])
    [x for x in np.argsort(predictions[2]) if x != np.argmax(predictions[0]) and x!= np.argmax(predictions[1])]
    np.argsort(predictions[3])
    [x for x in np.argsort(predictions[3]) if x != np.argmax(predictions[0]) and x != np.argmax(predictions[1]) and x != np.argmax(predictions[2])]
    np.argmax(predictions[0])
    np.argmax([x for x in predictions[1] if predictions[1].tolist().index(x) not in [np.argmax(predictions[0])]])
    np.argmax([x for x in predictions[2] if predictions[2].tolist().index(x) not in [np.argmax(predictions[0]), np.argmax(predictions[1])]])
    np.argmax([x for x in predictions[3] if predictions[3].tolist().index(x) not in [np.argmax(predictions[0]), np.argmax(predictions[1]), np.argmax(predictions[2])]])
    # (cont here!)[UPDATE: Need to use np.argsort to ensure none of the labels chosen result in one label being chosen more than once! Refer to this link for assistance: https://www.geeksforgeeks.org/python/how-to-get-the-n-largest-values-of-an-array-using-numpy/][Attempt to acheive solution is below] [NOTE: Basic idea is this: np.argmax(predictions[2][:np.argmax(predictions[1])])]
    # End of Attempt #2 At solution:
    test_stocks.loc[:,"Company"] = train_stocks.loc[:,"Company"].astype("str") #<-- Used to convert one-hot encoding back into strings interpretable by users.
    # Attempt #3 At solution:
    np.argsort(predictions[0]) #<-- NOTE: argsort sorts argument indices in ascending order! [UPDATE: There is also an edge case, if preds coincidentally are the same, then the next maximum should be pulled instead]
    np.argsort(predictions[1])
    np.argsort(predictions[2])
    np.argsort(predictions[3])
    test_stocks[test_stocks["Optimality"] == np.argmax(predictions[0])]["Company"]
    test_stocks[test_stocks["Optimality"] == [x for x in np.argsort(predictions[1]) if x != np.argsort(predictions[0])[-1]][-1]]["Company"]
    test_stocks[test_stocks["Optimality"] == [x for x in np.argsort(predictions[2]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] ][-1]]["Company"]
    test_stocks[test_stocks["Optimality"] == [x for x in np.argsort(predictions[3]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] and x != np.argsort(predictions[2])[-2]][0] ]["Company"]
    np.argmax(predictions[0])
    [x for x in np.argsort(predictions[1]) if x != np.argsort(predictions[0])[-1]][-1]
    [x for x in np.argsort(predictions[2]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] ][-1]
    [x for x in np.argsort(predictions[3]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] and x != np.argsort(predictions[2])[-2]]
    # UPDATE: Above works greatly when 4 companies are utilized. Now, focus shifts to programmatically adhering to situation for any finite amount of companies.
    conditions = []
    i = 1
    test_stocks.loc[:,"Company"] = train_stocks.loc[:,"Company"].astype("str") #<-- Used to convert one-hot encoding back into strings interpretable by users.
    print("--DEBUGGING CHECKPOINT: CHecking if algo works for any finite num of predictions---") # UPDATE: This process will be done after finishing up everything else.
    pb.set_trace()
    while i < (len(predictions)):
        # GOAL: a) Ensure that each prediction is unique, so a desired order can be implemented.
        lambda x: x != np.argsort(predictions[i - 1])[-1]
        conditions.append(lambda x: x != np.argsort(predictions[i - 1])[-1]) #<-- NOTE: Leaving x und   ef on purpose since it'll be defined in list comprehension below.
        # conditions.append(x != np.argsort(predictions[i - 1])[-1]) #<-- NOTE: Leaving x undef on purpose since it'll be defined in list comprehension below.
        # [x for x in np.argsort(predictions[1]) if x != np.argsort(predictions[0])[-1]][-1]
        print([x for x in np.argsort(predictions[i]) if conditions][-1])
        print("----")
        print(test_stocks[test_stocks["Optimality"] == [x for x in np.argsort(predictions[1]) if conditions][-1]]["Company"])
        print("----")
        i += 1
    # End of Attempt #3 At solution:
    """
    # NOTE: Above will reference ways to transform predictions into string format to be able to utilzied for user consumption
    ## Listing all data in history:
    print(train_history.history.keys())
    ## summarize train_history for accuracy
    plt.plot(train_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    integrationMode = True
    if(not integrationMode):
        plt.show()

    # plt.savefig("./ModelAccuracyPlot#1.png")
    # plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1.png")
    # print("--DEBUGGING CHECKPOINT: Investigating why plot doesn't save")
    # pb.set_trace() <-- Works as intended.
    # plt.savefig("../../../plots/ModelAccuracyPlot#1.png")
    plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1.png")
    # NOTE: Will replace True with booleans that will live at top of file
    # baseLinePerform, Experiment1, Experiment2, Experiment3 = True,True,True,True
    taskDesc = ["Baseline Performance","Experiment #1 Pictures","Experiment #2 Pictures","Experiment #3 Pictures"]
    # print("--DEBUGGING CHCECKECHPOINT:---")
    # pb.set_trace()
    if(baseLinePerform):
        print("---PRINTING THINGS FOR TASK: {taskDesc[0]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Baseline.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Baseline.png")
    elif(Experiment1):
        print("---PRINTING THINGS FOR TASK: {taskDesc[1]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Experiment#1.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Experiment#1.png")
    elif(Experiment2):
        print("---PRINTING THINGS FOR TASK: {taskDesc[2]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Experiment#2.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Experiment#2.png")
    elif(Experiment3):
        print("---PRINTING THINGS FOR TASK: {taskDesc[3]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Experiment#3.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Experiment#3.png")
    # PN: Will need to change number to prevent overriding. Will need some sort of boolean.
    ## summarize history for loss
    plt.plot(train_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    integrationMode = True
    if(not integrationMode):
        plt.show()

    # plt.savefig("./ModelLossPlot#1.jpg")
    # plt.savefig(f"{current_dir}/plots/ModelLossPlot#1.png")
    # plt.savefig("../../plots/ModelLossPlot#1.jpg")
    # plt.savefig(f"{current_dir}/plots/ModelLossPlot#1.png")
    if(baseLinePerform):
        print("---PRINTING THINGS FOR TASK: {taskDesc[0]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Baseline.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Baseline.png")
    elif(Experiment1):
        print("---PRINTING THINGS FOR TASK: {taskDesc[1]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Experiment#1.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Experiment#1.png")
    elif(Experiment2):
        print("---PRINTING THINGS FOR TASK: {taskDesc[2]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Experiment#2.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Experiment#2.png")
    elif(Experiment3):
        print("---PRINTING THINGS FOR TASK: {taskDesc[3]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Experiment#3.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Experiment#3.png")
    # PN: Will need to change number to prevent overriding. Will need some sort of boolean.
    # End of Body of plotting model
    # b) Printing Model Summary, which can be good to go into detail about:
    model.summary()
    # Body of saving model to a file to be used later
    # End of Body of saving model to a file to be used later
    # end of 6)
def ModelTrainingAndDevelopment():
    inNoteBook = False
    # Body of Neccessary Imports for Model Development.
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.utils import to_categorical, plot_model
    from tensorflow.keras.datasets import mnist
    # End of Body of Neccessary Imports for Model Development.
    # NOTE: Below is used to modify network parameters using experiment setup func(s).
    global x_train; global x_test
    demo = True
    parentDir = "C:/Users/adoct/Notes for CSCE Classes\\[Fall 2025\\]/Notes for CSCE 585/projectCode" if demo == False else os.getcwd().replace("\\","/")
    filePathToModelDir = f"{parentDir}/projectCode/MLLifecycle/ModelDevelopmentAndTraining/preparedDataset.csv" if inNoteBook == False else "preparedDataset.csv"
    train_stocks = pd.read_csv(f"{filePathToModelDir}"); train_labels = test_labels = train_stocks.loc[:,"Company"] #<-- References labels which are derived from custom engineered dataset.
    del train_stocks["Unnamed: 0"]
    test_stocks = pd.read_csv(f"{filePathToModelDir}"); test_labels = train_stocks.loc[:,"Company"] #<-- References labels which are derived from custom engineered dataset.
    del test_stocks["Unnamed: 0"]
    x_train = train_stocks.loc[:,train_stocks.columns != "Optimality"]
    y_train = train_labels
    x_test = test_stocks.loc[:,train_stocks.columns != "Optimality"]
    y_test = test_labels
    # ^^ Above ensures that prediction labels are y and x references the data used to make said decision.
    # "Compute the number of labels"
    num_labels = len(np.unique(y_train))
    # "Convert to one-hot vector"[we converted the labels to one-hot vectors using to_categorical]
    # y_train = to_categorical(y_train.index)
    # y_test = to_categorical(y_test.index)
    input_size = len(train_stocks.columns)  #<-- input_size refers to number of attributes for each row tuple of data.
    # Establishing Network Parameters
    global batch_size, hidden_units, dropout
    global model #<-- Have this here, so Experimental Setup functions can tweak model as needed.
    batch_size = 128
    hidden_units = 256
    dropout = 0.45
    isExperimentSetup3Active = False
    model = Sequential()
    def experimentSetup0():
       # This will reference the default settings irrespective to experiments.
       # 1) Creating the Model.
        print("---Undergoing Experiment Setup #0---")
        model.add(Input((x_train.shape[1],)))
        model.add(Dense(hidden_units))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units))
        model.add(Activation('relu')) if experimentSetup3() == None else experimentSetup3() #<-- This changes the Activation function, adhering to experimentSetup3!
        model.add(Dropout(dropout))
        model.add(Dense(num_labels))
        print("---End of Experiment Setup #0---")
        return
    def experimentSetup1():
        # Goal of exp: Want to see how model params affect the acuaracy of the model by modifying batch size and hidden units and dropout
        print("---Undergoing Experiment Setup #1---")
        setN = 1 #<-- Change number for this to get values from resp sets.
        experiment1Tuples: list[tuple] = [(128,256,0.45), (64,128,0.45), (128,256,0.3), ("""NOTE: Other tuples can change one or more parameters whilst keeping at least one constant""")]
        batch_size = experiment1Tuples[setN][0]
        hidden_units = experiment1Tuples[setN][1]
        dropout = experiment1Tuples[setN][2]
        print("---End of Experiment Setup #1---")
        return
    global listVerOfX_train
    listVerOfX_train = x_train.columns[:len(x_train.columns)-1].to_list()
    # ^^ Utilized to setup experiment #2 whose desc is below.
    def experimentSetup2(numQuantLvls = 2):
        # Goal of exp: Want to see model performance based on degree of quantanization of data
        print("---Undergoing Experiment Setup #2---")
        # NOTE: Below will involve replacing 255 with a different number based on degree of quantanization of data.
        # Using for loop to iterate through each column to apply this quantinization to each column.
        global x_train, x_test
        for i in listVerOfX_train:
            x_trainMax = x_train[i].max()
            x_trainMin = x_train[i].min()
            x_testMax = x_test[i].max()
            x_testMin = x_test[i].min()
            widthsOfQuant = [(x_trainMax - x_trainMin)/(numQuantLvls - 1),(x_testMax - x_testMin)/(numQuantLvls - 1)]
            x_train.loc[:,i] = x_train.loc[:,i]/widthsOfQuant[0]
            x_test.loc[:,i] = x_test.loc[:,i]/widthsOfQuant[0]
        print("---End of Experiment Setup #2---")
        return
    def experimentSetup3():
        # Goal of exp: Want to see model performance based on type of activation function from a subset of all possible activation functions.
        print("---Undergoing Experiment Setup #3---" if isExperimentSetup3Active else "---Experiment Setup #3 was skipped---")
        activationFuncs = ['elu', 'sigmoid', 'tanh' ]
        print("---End of Experiment Setup #3---" if isExperimentSetup3Active else "---End of Experiment Setup #3 was skipped---")
        return model.add(Activation(activationFuncs[0])) if isExperimentSetup3Active == True else None
    def experimentSetup4():
        print("---Undergoing Experiment Setup #4---")
        # Need to have an experiment that utilizes scaling analysis[in progress]
        print("---End of Experiment Setup #4---")
        return
    experimentSetup0()
    # experimentSetup1()
    experimentSetup2()
    # This is the output for one-hot vector, being sent into the softmax activation function.
    model.add(Activation('softmax'))
    model.summary()
    # 4) Defining the loss function and compiling model
    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
    # 5) Training the network
    x_train.loc[:,"Company"] = pd.Series([2**i for i in range(x_train.shape[0])])
    x_test.loc[:,"Company"] = pd.Series([2**i for i in range(x_train.shape[0])])
    x_train = x_train.astype("float64")
    x_test = x_test.astype("float64")
    x_train.loc[:,"Company"] = x_train.loc[:,"Company"].astype("int32")
    x_test.loc[:,"Company"] = x_test.loc[:,"Company"].astype("int32")
    x_trainCopy = tf.data.Dataset.from_tensor_slices((x_train.values.astype(np.float32),tf.convert_to_tensor(y_train).numpy().astype(np.float32)))
    x_testCopy = tf.data.Dataset.from_tensor_slices((x_test.values.astype(np.float32),tf.convert_to_tensor(y_test).numpy().astype(np.float32)))
    train_history = model.fit(x_train,y_train,epochs=20, batch_size=batch_size)
    # 6) Evaluating Model
    acc = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc[1]))
    
    # Body of making copy of test_stocks to be used later
    test_stocksCopy = test_stocks

    # End of Body of making copy of test_stocks to be used later

    test_stocks["Company"] = test_stocks.index; test_stocks.loc[:,"Company"] = test_stocks.loc[:,"Company"].astype("int32")
    # 6) Verifying and Visualzing the Predictions
    # a) Obtaining accuarrcy of the predictions:
    WriteModelToAFile = True
    predictions = model.predict(x_test) #<-- NOTE: This retunrns an array of probabilities for each class. Thus, for future consumption by person, could assign these predictions to a column added to test_stocks?
    # Write code for writing model to a file here
    local = False #<-- '""' Needs to refer to virtual environment. [VIRTUAL ENVIRONMENT ADDRESS THING[NOTE]: Will need to change this file path to adhere to virtual environment!] 
    ImprovedAlgo = True
    if(WriteModelToAFile):
        # Goal: Make copy of test stocks, add predictions column referencing model's 
        print("---Writing Model to a file for future use---")
        # print("---DEBUGGING CHECKPOINT: Writing Model to a file for future use---")
        # pb.set_trace()
        # ^^ ABOVE NOT NEEDED!
        
        copyTwoSendOff = test_stocks
        # copyTwoSendOff["Model Predictions"] = predictions
        # copyTwoSendOff.loc[:,"Model Predictions"] = predictions
        # copyTwoSendOff.loc[:,"Model Predictions"] = pd.Series([np.argmax(x) for x in predictions])
        # Body of modifying predictions to ensure that each prediction results in only ONE thing being chosen
        # Attempt #1[closest I've gotten]
        list_a = test_stocks["Optimality"].tolist()
        list_b = [np.argsort(predictions[i])[-1] for i in range(len(predictions)) if i < len(predictions) - 1 and np.argsort(predictions[i][-1]) in np.argsort(predictions[i+1][-1])]
        list_b = [x.item() for x in list_b]
        missing_element = set(list_a).difference(set(list_b)) #<-- equal to list_a \cap (list_b)^c . 
        # End of Attempt #1

        # Attempt #2
        testerU = [np.argsort(x) for x in predictions]
        # Algorithm for ensuring unique signings for labels:
        labelResults = ["" for i in range(len(predictions))]

        # boolValues = list(map(lambda x,i: x != labelResults[i], ))
        # Body of creating array that'll be dynamically added to for all i (x != labelResults[i]) .
        def boolPP(a,b):
            return a != b
        booVals = [lambda x,y: x != y]
        labelResults[0] = testerU[0][0]
        for i in range(len(labelResults) - 1):
            labelResults[i+1] = [x for x in testerU[i] if x not in labelResults[:i+1]][0]

        # End of algorithm[ALGORITHM WORKS!!]
        
        # End of Attempt #2
        copyTwoSendOff["Model Predictions"] = pd.Series([np.argmax(x) for x in predictions]) if ImprovedAlgo == False else pd.Series(labelResults) 
        # thoughts. 
        # print("--DEBUGGING CHECKPOINT: Sorting predictions in order so they reflect--- ")
        # pb.set_trace()
        # End of Attempt #1
        # End of Body of modifying predictions to ensure that each prediction results in only ONE thing being chosen
        copyTwoSendOff.loc[:, "Company"] = test_labels

        
        demo = True
        parentDir = "C:/Users/adoct/Notes for CSCE Classes\\[Fall 2025\\]/Notes for CSCE 585/projectCode" if demo == False else os.getcwd().replace("\\","/")
        writePathForModelPreds = f"{parentDir}/projectCode/MLLifecycle/ModelDevelopmentAndTraining/ModelPredictions.csv" if local == True else "ModelPredictions.csv" #<-- '""' Needs to refer to virtual environment. 

        
        
        
        copyTwoSendOff.to_csv(f"{writePathForModelPreds}")
        print("---End of Writing Model to a file for future use---")

        
        
        
        # End of writing code for writing model to a file here
    predictions[0]

    # print("---DEBUGGING CHECKPOINT: Improving Predictions Algorithm---")
    # pb.set_trace()
    # NOTE: Above will reference ways to transform predictions into string format to be able to utilzied for user consumption
    ## Listing all data in history:
    print(train_history.history.keys())
    ## summarize train_history for accuracy
    plt.plot(train_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    integrationMode = True
    if(not integrationMode):
        plt.show()

    # plt.savefig("./ModelAccuracyPlot#1.png")
    # plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1.png")
    # print("--DEBUGGING CHECKPOINT: Investigating why plot doesn't save")
    # pb.set_trace() <-- Works as intended.
    # plt.savefig("../../../plots/ModelAccuracyPlot#1.png")
    plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1.png")
    # NOTE: Will replace True with booleans that will live at top of file
    # baseLinePerform, Experiment1, Experiment2, Experiment3 = True,True,True,True
    taskDesc = ["Baseline Performance","Experiment #1 Pictures","Experiment #2 Pictures","Experiment #3 Pictures"]
    # print("--DEBUGGING CHCECKECHPOINT:---")
    # pb.set_trace()
    if(baseLinePerform):
        print("---PRINTING THINGS FOR TASK: {taskDesc[0]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Baseline.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Baseline.png")
    elif(Experiment1):
        print("---PRINTING THINGS FOR TASK: {taskDesc[1]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Experiment#1.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Experiment#1.png")
    elif(Experiment2):
        print("---PRINTING THINGS FOR TASK: {taskDesc[2]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Experiment#2.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Experiment#2.png")
    elif(Experiment3):
        print("---PRINTING THINGS FOR TASK: {taskDesc[3]}---")
        plt.savefig(f"{current_dir}/plots/ModelAccuracyPlot#1_Experiment#3.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelAccuracyPlot#1_Experiment#3.png")
    # PN: Will need to change number to prevent overriding. Will need some sort of boolean.
    ## summarize history for loss
    plt.plot(train_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    integrationMode = True
    if(not integrationMode):
        plt.show()

    # plt.savefig("./ModelLossPlot#1.jpg")
    # plt.savefig(f"{current_dir}/plots/ModelLossPlot#1.png")
    # plt.savefig("../../plots/ModelLossPlot#1.jpg")
    # plt.savefig(f"{current_dir}/plots/ModelLossPlot#1.png")
    if(baseLinePerform):
        print("---PRINTING THINGS FOR TASK: {taskDesc[0]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Baseline.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Baseline.png")
    elif(Experiment1):
        print("---PRINTING THINGS FOR TASK: {taskDesc[1]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Experiment#1.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Experiment#1.png")
    elif(Experiment2):
        print("---PRINTING THINGS FOR TASK: {taskDesc[2]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Experiment#2.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Experiment#2.png")
    elif(Experiment3):
        print("---PRINTING THINGS FOR TASK: {taskDesc[3]}---")
        plt.savefig(f"{current_dir}/plots/ModelLossPlot#1_Experiment#3.png")
        plt.savefig(f"{sys.path[3]}/plots/ModelLossPlot#1_Experiment#3.png")
    # PN: Will need to change number to prevent overriding. Will need some sort of boolean.
    # End of Body of plotting model
    # b) Printing Model Summary, which can be good to go into detail about:
    model.summary()
    # Body of creating transition matrices and saving them to file(s) for future use
    # print("---DEBUGGING CHECKPOINT: Figuring out how to print out confusion matrix---")
    # pb.set_trace()
    """    
      cf = tf.math.confusion_matrix(
        labels=[max(2^i) for i in y_test],
        predictions=predictions,
        num_classes=num_labels,
    )
    # print(cf)
    """    
    # End of creating transition matrices and saving them to file(s) for future use
    # end of 6)
    
attempt3()
# ModelTrainingAndDevelopment()
