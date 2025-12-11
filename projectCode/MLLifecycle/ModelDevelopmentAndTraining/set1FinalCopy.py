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
            isExperimentSetup3_newModelActive = False
            global x_train, x_test
            global y_train, y_test, model
            global threshold
            # NOTE: Insert body of newModel here!
            def experimentSetup0_newModel():
                # Body of loading in data for classifiers to consume to make predictions
                print("---Loading in data for classifier consumption---")

                
                print("---End of Loading in data for classifier consumption---")
                # End of Body of loading in data for classifiers to consume to make predictions
                # Body of creating classifiers to be used by model
                print("---Creating Classifiers to for model choices---")
                from sklearn.linear_model import SGDClassifier #<-- NOTE: Can use this in place of model??
                
                sgd_clf = SGDClassifier(loss="log_loss",random_state=42)
                # NOTE: Between this, need to have: i) Model needs to be trained on optimality, ii) Model needs to be in able to apply optimality number to each company given, iii) company labels need to be one-hot encoded or transformed into integers for model intepretation. 

                global x_train
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
                        x_train[i].fillna(x_train[i].mean(), inplace=True)
                
                x_test["Company"] = pd.Series([i for i in range(len(x_test["Company"]))])
                if(x_test.isna().any(axis = 1).sum()):
                    for i in x_test.columns:
                        x_test[i].fillna(x_test[i].mean(), inplace=True)
                sgd_clf.fit(x_train, x_train["Company"]) #<-- This command is used to train the model
                
                
                # sgd_clf.predict(y_test); #<-- This command is used to TEST/evalutate the model. 
                global model
                model = sgd_clf.predict(x_test); #<-- NOTE: x_test is used here b/c scikit-learn automatically omitts the label column.
                
                
                
                
                
                
                
                print("---End of Creating Classifiers to for model choices---")
                # End of Body of creating classifiers to be used by model


                # Body of evaluating classifier's results via plotting etc
                print("---Evaluating Classifier's Results---")
                pb.set_trace() 
                
                from sklearn.model_selection import cross_val_score
                # cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                y_train = x_train["Company"]
                y_train = pd.concat([y_train, y_train], axis=0)
                x_train = pd.concat([x_train, x_train], axis=0)
                cross_val_score(sgd_clf, x_train, y_train, cv=2, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                from sklearn.model_selection import cross_val_score
                cross_val_score(sgd_clf, x_train, y_train, cv=2, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                from sklearn.model_selection import cross_val_predict
                y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=2)
                from sklearn.metrics import confusion_matrix
                confusion_matrix(y_train, y_train_pred)
                y_scores = cross_val_predict(sgd_clf, x_train, y_train, cv=2, method="decision_function")

                # NOTE: Everything above this point works! Only problem is getting plotting to work!
                from sklearn.metrics import precision_recall_curve

                print("---Need to work on creating precision curve---")
                pb.set_trace()
                """
                precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores);
                
                # ^^
                # Correct version:

                from sklearn.preprocessing import label_binarize

                classes = list(set(y_train)) 

                # "Binarize true labels"
                y_bin = label_binarize(y_train, classes=classes)

                # "y_scores must be shape (n_sampls, n_classes)"

                proba = sgd_clf.predict_proba(x_train)
                for i,cls in enumerate(classes):
                    precision,recall, thresholds = precision_recall_curve(y_bin[:,i], proba[:,i])
                
                
                """
                print("---End of Need to work on creating precision curve---")
                from sklearn.metrics import precision_score, recall_score
                precision_score(y_train, y_train_pred, average="micro") # == 4096/(4096*1522) = \text{precision} = \frac{TP}{TP + FP}
                recall_score(y_train, y_train_pred, average="micro") # == 4096/(4096*1522) = \text{recall} = \frac{TP}{TP + FN}
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
    global model #<-- Have this here, so Experimental Setup functions can tweak model as needed.
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
            isExperimentSetup3_newModelActive = False
            global x_train, x_test
            global y_train, y_test, model
            global threshold
            # NOTE: Insert body of newModel here!
            def experimentSetup0_newModel():
                # Body of loading in data for classifiers to consume to make predictions
                print("---Loading in data for classifier consumption---")

                
                exp1Active = False
                exp3Active = False
                if (exp3Active == True):
                    experimentSetup3_newModel(pair1=True)
                    # experimentSetup3_newModel(pair1=False)
                print("---End of Loading in data for classifier consumption---")
                # End of Body of loading in data for classifiers to consume to make predictions
                # Body of creating classifiers to be used by model
                print("---Creating Classifiers to for model choices---")
                from sklearn.linear_model import SGDClassifier #<-- NOTE: Can use this in place of model??
                
                sgd_clf = SGDClassifier(loss="log_loss",random_state=42)
                # NOTE: Between this, need to have: i) Model needs to be trained on optimality, ii) Model needs to be in able to apply optimality number to each company given, iii) company labels need to be one-hot encoded or transformed into integers for model intepretation. 

                global x_train
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
                        x_train[i].fillna(x_train[i].mean(), inplace=True)
                
                x_test["Company"] = pd.Series([i for i in range(len(x_test["Company"]))])
                if(x_test.isna().any(axis = 1).sum()):
                    for i in x_test.columns:
                        x_test[i].fillna(x_test[i].mean(), inplace=True)
                sgd_clf.fit(x_train, x_train["Company"]) #<-- This command is used to train the model
                
                
                # sgd_clf.predict(y_test); #<-- This command is used to TEST/evalutate the model. 
                sgd_clf.predict(x_test); #<-- NOTE: x_test is used here b/c scikit-learn automatically omitts the label column.
                
                
                
                
                
                
                
                print("---End of Creating Classifiers to for model choices---")
                # End of Body of creating classifiers to be used by model


                # Body of evaluating classifier's results via plotting etc
                print("---Evaluating Classifier's Results---")
                pb.set_trace() 
                
                from sklearn.model_selection import cross_val_score
                # cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                y_train = x_train["Company"]
                y_train = pd.concat([y_train, y_train], axis=0)
                x_train = pd.concat([x_train, x_train], axis=0)
                cross_val_score(sgd_clf, x_train, y_train, cv=2, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                from sklearn.model_selection import cross_val_score
                cross_val_score(sgd_clf, x_train, y_train, cv=2, scoring="accuracy") #<-- this returns the classifer model's accuracy in percentage form
                from sklearn.model_selection import cross_val_predict
                y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=2)
                from sklearn.metrics import confusion_matrix
                confusion_matrix(y_train, y_train_pred)
                conf_mx = confusion_matrix(y_train,y_train_pred)
                conf_mx
                # ^^ Use following code for plotting confusion matrix!
                conf_mx = confusion_matrix(y_train,y_train_pred)
                conf_mx

                plt.matshow(conf_mx, cmap=plt.cm.gray)
                plt.show()

                
                # End of Use following code for plotting confusion matrix!
                y_scores = cross_val_predict(sgd_clf, x_train, y_train, cv=2, method="decision_function")

                # NOTE: Everything above this point works! Only problem is getting plotting to work!
                from sklearn.metrics import precision_recall_curve, average_precision_score

                print("---Need to work on creating precision curve---")
                pb.set_trace()
                """
                precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores);
                
                """
                # ^^
                # Correct version:

                from sklearn.preprocessing import label_binarize

                classes = list(set(y_train)) 

                # "Binarize true labels"
                y_bin = label_binarize(y_train, classes=classes)

                # "y_scores must be shape (n_sampls, n_classes)"

                proba = sgd_clf.predict_proba(x_train)
                precision,recall, thresholds = "", "", ""
                for i,cls in enumerate(classes):
                    precision,recall, thresholds = precision_recall_curve(y_bin[:,i], proba[:,i])   
                    ap = average_precision_score(y_bin[:, i], proba[:, i])
                    plt.plot(recall, precision, label=f"Class {classes[i]} (AP = {ap:.2f})")
                
                """
                CURVE COMPELTE!!!! JUST NEED TO PLOT!
                """
                
                print("---End of Need to work on creating precision curve---")
                from sklearn.metrics import precision_score, recall_score
                precision_score(y_train, y_train_pred, average="micro") # == 4096/(4096*1522) = \text{precision} = \frac{TP}{TP + FP}
                recall_score(y_train, y_train_pred, average="micro") # == 4096/(4096*1522) = \text{recall} = \frac{TP}{TP + FN}

                from sklearn.metrics import f1_score
                f1_score(y_train, y_train_pred, average="micro") 

                y_scores = sgd_clf.decision_function(x_test)
                y_scores

                if(exp1Active == True):
                    experimentSetup1_newModel()
                   
                threshold = 0
                y_scores_thres = (y_scores > threshold)

                # Increasing Threshold
                # threshold = 8000
                # y_scores_thres = (y_scores > threshold)


                # y_scores = cross_val_predict(sgd_clf, x_train, y_train, cv=2, method="decision_function")

                from sklearn.metrics import precision_recall_curve
                # NOTE: This portion won't work until I change it to support multiclass Precision-Recall version
                pb.set_trace()
                precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores);

                def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
                    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
                    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
                    plt.show()

                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

                # plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
                threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)] # == 7813 
                """
                """


                precision_score(y_train, y_train_pred, average="micro")
                recall_score(y_train, y_train_pred, average="micro")

                """
                import matplotlib.pyplot as plt
                from sklearn.metrics import roc_curve, auc
                from sklearn.preprocessing import label_binarize
                import numpy as np

                from sklearn.preprocessing import label_binarize

                classes = list(set(y_train)) 

                # "Binarize true labels"
                y_bin = label_binarize(y_train, classes=classes)

                # "y_scores must be shape (n_sampls, n_classes)"

                proba = sgd_clf.predict_proba(x_train)

                plt.figure(figsize=(8,6))

                for i, cls in enumerate(classes):
                    fpr,tpr, _ = roc_curve(y_bin[:,i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

                plt.plot([0,1], [0,1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Multiclass ROC Curve (OvR)")
                plt.legend()
                plt.show()
                fpr, tpr, thresholds = roc_curve(y_train, y_scores)

                def plot_roc_curve(fpr, tpr, label=None):
                    plt.plot(fpr,tpr,linewidth=2,label=label)
                    plt.plot([0,1], [0,1], 'k--') # "Dashed diagonal"
                            # [...] #<-- "Add axis labels and grid"
                    plot_roc_curve(fpr,tpr)
                    plt.show()

                

                """
                # from sklearn.metrics import roc_auc_score

                # roc_auc_score(y_train, y_scores, average='micro')


                    
                print("---End of Evaluating Classifier's Results---")

                
                
                # End of Body of evaluating classifier's results via plotting etc

                # Body of tweaking the precison/recall tradeoff to make the perfect classifier model
                print("---Tweaking precision/recall tradeoff to make perfect classifier---")



                
                print("---End of Tweaking precision/recall tradeoff to make perfect classifier---")

                
                
                # End of Body of tweaking the precison/recall tradeoff to make the perfect classifier model
                print("---End of Experiment Setup #0_newModel---")
                return
            def experimentSetup1_newModel(pair1 = True):
                    # Goal of exp: Want to see how model params affect the acuaracy of the model by modifying batch size and hidden units and dropout[UPDATE: Instead of these three params, will use threshold, precision, and recall instead!]
                    print("---Undergoing Experiment Setup #1_newModel---")
                    pb.set_trace()
                    # NOTE: Will set global variable to threshold
                    experiment1Vals = [100, 1000]
                    global threshold
                    if(pair1):
                        threshold = experiment1Vals[0]
                    else:
                        threshold = experiment1Vals[1]
                    
                    # NOTE: Need to replace metrics above with classifier parameters.[cmoplete, only one is merely threshold] Need to do more invesigating. 
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
                global x_test, x_train
                # Goal of exp: Will be responsible for removing feature-pairs[complete]
                print("---Undergoing Experiment Setup #3_newModel---")
                if(pair1):
                    del x_test[['P/E', 'P/B']]
                    del x_train[['P/E', 'P/B']]
                else:
                    del x_test[['NCAV', 'Share Price']]
                    del x_train[['NCAV', 'Share Price']]
                print("---End of Experiment Setup #3_newModel---" if isExperimentSetup3_newModelActive else "---End of Experiment Setup #3_newModel was skipped---")
                return
            
            experimentSetup0_newModel()
            # experimentSetup1_newModel()
            # experimentSetup2_newModel()
             
    # newModelIntegration(False)
    # pb.set_trace()
    newModelIntegration(True)

    
    # 4) Defining the loss function and compiling model

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
    
    # Body of making copy of test_stocks to be used later
    test_stocksCopy = test_stocks

    # End of Body of making copy of test_stocks to be used later

    test_stocks["Company"] = test_stocks.index; test_stocks.loc[:,"Company"] = test_stocks.loc[:,"Company"].astype("int32")
    # 6) Verifying and Visualzing the Predictions
    # a) Obtaining accuarrcy of the predictions:
    WriteModelToAFile = True
    predictions = model.predict(x_test) #<-- NOTE: This retunrns an array of probabilities for each class. 
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
attempt3()
# ModelTrainingAndDevelopment()