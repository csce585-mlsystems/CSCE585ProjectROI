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
# Body of modes of plots
baseLinePerform, Experiment1, Experiment2, Experiment3 = True,False,False,False #<-- Mode #1
# baseLinePerform, Experiment1, Experiment2, Experiment3 = False,True,False,False #<-- Mode #2
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
    y_train = to_categorical(y_train.index)
    y_test = to_categorical(y_test.index)
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
    y_train = to_categorical(y_train.index)
    y_test = to_categorical(y_test.index)
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
        print("--DEBUGGING CHECKPOINT: Retrying to make algorithm for unique label selection--- ")
        pb.set_trace()
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

    # print("---DEBUGGING CHECKPOINT: Improving Predictions Algorithm---")
    # pb.set_trace()
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
    
# attempt3()
