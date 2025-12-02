# NOTE: As of 11/30/25, MODEL DEV AND TRAINING IS COMPELTE! Need to transfer this to 
# NOTE: In regards to testing out program, use following command. Refer to its comment for reference: $ find ProjectCode/components ProjectCode/routes -name "*.py" | xargs --delimiter="\n" grep -E "^(import|from)" #<-- NOTE: Use this code to find the files to: a) determine neccessary downloads for pip, and b) determine where to call model and send output etc from. 

# Purpose: This file will contain code that helps with Model Development. 

# Body of neccessary imports
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np #<-- May be optional not sure as of 10/13/25. 
import pdb as pb
# import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist


# end of body of neccessary imports

# Important Steps in Model Development: 1) Obtaining the training data, 2) Create the model containing initialized weights and a bais[which would be very involved with using numbers from certain columns], 3) Observe model's performance before training, 4) Defining a loss function for model, 5) Write a basic Training Loop
def attempt1():
    # Section 1
    # Purpose of section: 1) Obtaining the training data

    # end of purpose of section

    # Psuedosteps for exercising purpose of section
    # - Need to make sure data is clean
    # - Need to have an object that references clean dataset
    # - Training data 
    # end of psuedosteps for exercising purpose of section 

    # body of section 1



    # end of body of section 1
    # End of Section 1


    # Section 2
    # Purpose of section: 2) Create the model containing initialized weights and bias

    # end of purpose of section

    # Psuedosteps for exercising purpose of section

    # end of psuedosteps for exercising purpose of section 

    # body of section 2
    # NOTE: Code is subject to change, just acting as starting point right now.
    class Model(tf.Module):
        def __init__(self):
            # "Randomly genearte weight and bias terms"
            rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed = 22)
            # "Initialize Model Parameters"
            self.w_q = tf.Variable(rand_init[0])
            self.w_l = tf.Variable(rand_init[1])
            self.b = tf.Variable(rand_init[2])
        @tf.function
        def __call__(self,x):
            # "Quadratic Model: quadratic_weight * x^2 + linear_weight*x + bias"
            return self.w_q * (x**2) + self.w_l * x + self.b

    # end of body of section 2

    # End of Section 2


    # Section 3
    # Purpose of section:  3) Observe model's performance before training

    # end of purpose of section

    # Psuedosteps for exercising purpose of section

    # end of psuedosteps for exercising purpose of section 

    # body of section 3

    # NOTE: Code is subject to change, just acting as starting point right now.
    quad_model = Model() #<-- instantiation of model
    def plot_preds(x,y,f,model,title):
       plt.figure() 
       plt.plot(x,y, '.', label='Data')
       plt.plot(x,f(x), label='Ground Truth')
       plt.plot(x,model(x), label='Predictions')
       plt.title(title)
       plt.legend()

    plot_preds(x,y,f,quad_model,'Before Tranining')



    # End of body of section 3
    # End of Section 3


    # Section 4
    # Purpose of section: 4) Defining the loss function

    # end of purpose of section

    # Psuedosteps for exercising purpose of section

    # end of psuedosteps for exercising purpose of section 

    # End of Section 4


    # Section 5
    # Purpose of section: 5) Write a basic Training Loop

    # end of purpose of section

    # Psuedosteps for exercising purpose of section

    # end of psuedosteps for exercising purpose of section 

    # Body of Section 5
    batch_size = 32
    # NOTE: This may be a good reference for sending in clean training data for model to use. 
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.shuffle(buffer_size=x.shape[0].batch(batch_size))

    # a) "Set Training Parameters"
    epochs = 100
    learning_rate = 1e-2
    losses= []

    # b) "Format Training Loop"
    for epoch in range(epochs):
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                batch_loss = mse_loss(quad_model(x_batch),y_batch) 
            # 1) "Update parameters with respect to the gradient calculations"
            grads = tape.gradient(batch_loss,quad_model.variables)
            for g,v in zip(grads, quad_model.variables):
                v.assign_sub(learning_rate*g)
        # 2) "Keep track of model loss per epoch"
        loss = mse_loss(quad_model(x), y)
        losses.append(loss)
        if epoch %10 == 0:
            print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')

    # c) "Plot Model Results"
    print("\n")
    plt.plot(range(epochs), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE loss vs Training Iterations");

    plt.show()
    # Now, observe your model's performacne after training: 
    plot_preds(x,y,f,quad_model, 'After Training')
    # End of Body of Section 5
    # End of Section 5


    ## NOTE: Below references an alternative approach. This ranges from getting data to providing statistcis for model eval. Does not include data visualization portion. 
"""
# Attempt 2
# Modificaitons needed: a) Need to find place where to put modified dataset, b) Need to ensure that a training data set is created from the dataset(s) that we have[need to make sure an additional row or column is provided for the classification], (cont here if applicable) 
# TensorFlow and tf.keras
import tensorflow as tf

# "Helper Libraries"
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 1) Import Dataset
#fashion_mnist = tf.keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
filepathToTrainingSet: str
filepathToTestSet: str
filePathToTrainingSet += f"/testSetStrat{1}.py" #<-- number will change based on strat being used.
train_stocks = pd.read_csv(f"{filepathToTrainingSet}); train_labels = test_labels = ["will reference company names"];
test_stocks = pd.read_csv(f"{filepathToTrainingSet}); test_labels = ["will reference company names"];

# end of 1)
# 2) Provide class names for ML Model to make predictions
#class_names = ['T-shirt/top', 'Trourser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
 class_names = ['Apple', 'Google', 'Amazon']

# end of 2)

# 3) Exploring the data
len(train_labels)

test_images.shape


len(test_labels)


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#plt.show()
# end of 3)
# 4) Creating the Model[can be created using keras OR created manually by inherting the tf.Module object][aka the Neural Network][UPDATE: Think it'd make sense to create aa FNN first using things below]
model = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=(28,28)),
tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']) #<-- NOTE: Metrics here, can probably be modified[UPDATE: No need, accuracy is what is important right now]

# end of 4)
# 5) Evaluating the Model
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=2)

print('\nTest accuracy:', test_acc)


probability_model = tf.keras.Sequential([model, 
tf.keras.layers.Softmax()])

# a) Obtaining the accuarcy of the predictions
predictions = probability_model.predict(test_images)

predictions[0]


np.argmax(predictions[0]) #<-- Returns the prediction that Neural Network was most comfortable with. 

# 6) Verifying and Visualzing the Predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'blue'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predicitons_array), class_names[true_label], color=color)

def plot_image(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#7777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions[i],test_labels,test_images)
plt.subplot(1,2,2)
plot_image(i,predictions[i],test_labels)
plt.show()

# end of 6)

"""
def attempt3(): #<-- NOTE: This attempt is what will be used for Model Portion. 
    inNoteBook = False
    
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.utils import to_categorical, plot_model
    from tensorflow.keras.datasets import mnist
    # NOTE: Make sure to NOT worry about imports. 
    # NOTE: Below is used to modify network parameters using experiment setup func(s). 
    global x_train; global x_test

    # "load mnist dataset"
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # filePathToTrainingSetDir: str = "MLLifecycle/ModelDevelopment/TrainingSets" #<-- fill in later[complete] UPDATE as of 11/10/25: Replaced since I am using script.py for running pipeline. 
    """     
filePathToTrainingSetDir: str = "./ModelDevelopment/TrainingSets" #<-- fill in later[complete]
    filepathToTrainingSet: str = filePathToTrainingSetDir
    filePathToTrainingSet += f"/trainingSetStrat{1}.py" #<-- number will change based on strat being used.
    filePathToTestSetDir: str = "MLLifecycle/ModelDevelopment/TestingSets" #<-- fill in later[complete]
    filepathToTestSet: str = filePathToTestSetDir
    filePathToTestSet += f"/testSetStrat{1}.py" #<-- number will change based on strat being used.
    """    
    filePathToModelDir = "C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/ModelDevelopment/preparedDataset.csv" if inNoteBook == False else "preparedDataset.csv"
    # print("---DEBUGGING CHECKPOINT #1: Ensuring that train_stocks & train_labels reference the right things---")
    # pb.set_trace() [COMPLETE]
    # train_stocks = pd.read_csv(f"{filePathToModelDir}"); train_labels = test_labels = train_stocks.loc["Company"] or ["will reference company names"];#<-- References labels which are derived from custom engineered dataset. 
    train_stocks = pd.read_csv(f"{filePathToModelDir}"); train_labels = test_labels = train_stocks.loc[:,"Company"] #<-- References labels which are derived from custom engineered dataset. 
    del train_stocks["Unnamed: 0"]
    test_stocks = pd.read_csv(f"{filePathToModelDir}"); test_labels = train_stocks.loc[:,"Company"] # or ["will reference company names"]; #<-- References labels which are derived from custom engineered dataset. 
    del test_stocks["Unnamed: 0"]
    x_train = train_stocks.loc[:,train_stocks.columns != "Optimality"]
    y_train = train_labels
    x_test = test_stocks.loc[:,train_stocks.columns != "Optimality"]
    y_test = test_labels
    # ^^ Above ensures that prediction labels are y and x references the data used to make said decision. [UPDATE: Due to this, will need to make a change in data prep folder]
            

    # "Compute the number of labels"
    num_labels = len(np.unique(y_train))

    # "Convert to one-hot vector"[we converted the labels to one-hot vectors using to_categorical]

    # resultantDataFrame = pd.concat([pd.DataFrame(x) for x in listOfSeriesToCreateDataFrame]).reset_index() #<-- used list comprehension to transform listOfSeries to resultantDataFrame. 
    # print("---DEBUGGING CHECKPOINT #1: Ensuring that train_stocks & train_labels reference the right things---")
    # pb.set_trace() [COMPLETE!]
    # y_train = to_categorical([range(y_train.shape[0]) for x in y_train])
    # y_test = to_categorical([ range(y_test.shape[0]) for x in y_test])
    y_train = to_categorical(y_train.index)
    y_test = to_categorical(y_test.index)
    """
    NOTE: Below is NOT needed
    listOfHotEncodings = list(range(y_train.shape[0]))
    # y_train = to_categorical([ dict[x] for x in y_train])
    for i in range(y_train.shape[0]):
        # listOfHotEncodings[i] = 2**(y_train.shape[0]) - 2**i #<-- Uses one hot encoding by using decimal value that refs binary value. 1 = 0001, 2 = 0010, 4 = 0100, 8 = 1000 and so forth. 
        listOfHotEncodings[i] = 2**i #<-- Uses one hot encoding by using decimal value that refs binary value. 1 = 0001, 2 = 0010, 4 = 0100, 8 = 1000 and so forth. 

    y_train = listOfHotEncodings
    # y_test = to_categorical([ range(y_test.shape[0]) for x in y_test])
    for i in range(y_test.shape[0]):
        # listOfHotEncodings[i] = 2**(y_test.shape[0]) - 2**i #<-- Uses one hot encoding by using decimal value that refs binary value. 1 = 0001, 2 = 0010, 4 = 0100, 8 = 1000 and so forth. 
        listOfHotEncodings[i] = 2**i #<-- Uses one hot encoding by using decimal value that refs binary value. 1 = 0001, 2 = 0010, 4 = 0100, 8 = 1000 and so forth. 
    y_test = listOfHotEncodings
    """
    
    # NOTE: Currently here as far as debugging. [ALSO, whilst working make sure that you know that: NOTE: Need to find a dataset that references ranked optimality of stocks based on value investing strategy to use as a benchmark for model results. ]

    # ----May snippet below may be optional?---
    # "image dimensions(assumed square)"
    # image_size = x_train.shape[1]
    # input_size = image_size * image_size
    input_size = len(train_stocks.columns) # #<-- UPDATE: input_size is NOT optional. Set this assuming that input_dim is number of dims/attributes for each feature which refers to number of columns.  However, I believe this is 

    # "Resize and Normalize"
    """ 
    x_train = np.reshape(x_train, [-1,input_size])
    x_train = x_train.astype('float32')/255
    x_test = np.reshape(x_test, [-1,input_size])
    x_test = x_test.astype('float32')/255
    """
    # ----May snippet above may be optional?[UPDATE: Above is required, need to come up with normalization process]---
    # "Network Parameters"
    # NOTE: The following will reference a list of tuple(s) that will be used to facilitate first experiment. 
    listOfExperimentSetupFuncs = ["""Plan to insert functions for doing experiment(s) setup here"""]
    global batch_size, hidden_units, dropout
    global model #<-- Have this here, so experimental setup functions can tweak model as needed. 
    batch_size = 128
    # batch_size = experiment1Tuples[0][0] 
    hidden_units = 256
    # hidden_units = experiment1Tuples[0][1]
    dropout = 0.45
    # dropout  = experiment1Tuples[0][2]
    isExperimentSetup3Active = False
    # model = Sequential()
    model = Sequential()
    def experimentSetup0():
       # This will reference the default settings irrespective to experiments. 
        print("---Undergoing Experiment Setup #0---")
        # model.add(Dense(hidden_units,input_dim=input_size))
        # model.add(Dense(hidden_units,input_shape=(input_size-1,)))
        # print("---DEBUGGING CHECKPOINT: Checking model's reactions---")
        # pb.set_trace()
        # model.add(Input((input_size,)))
        model.add(Input((x_train.shape[1],)))
        # model.add(Dense(hidden_units,input_shape=(input_size,)))
        model.add(Dense(hidden_units))
        # NOTE: Above is causing following error: "ValueError: Exception encountered when calling Sequential.call(). Invalid input shape for input Tensor("data:0", shape=(5,), dtype=float32). Expected shape (None, 6), but input has incompatible shape (5,)" [UPDATE: Error is originating from fact that x_trainCopy and x_testCopy 's shapes are (5,) and (5,)]
        model.add(Activation('relu')) #<-- This is used to add activation function to model[UPDATE: May need to replace Activation('relu') by making them default...not sure to facilitate experimentSetup3 OR I can simply see what happens when the 2nd to LAST acivation is changed] 
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units))
        model.add(Activation('relu')) if experimentSetup3() == None else experimentSetup3() #<-- This changes the Activation function, adhering to experimentSetup0![Thus, as of 11/23/25: There are two experiments ready to go for Milestone 1]
        model.add(Dropout(dropout))
        model.add(Dense(num_labels))
        print("---End of Experiment Setup #0---")
        return

    def experimentSetup1():
        # Goal of exp: Want to see how model params affect the acuaracy of the model by modifying batch size and hidden units and dropout[pn: make sure to have a subset of the powerset of params be modified in this exp. ]
        print("---Undergoing Experiment Setup #1---")
        print("---DEBUGGING CHECKPOINT #6: Testing out experimentSetup1---")
        pb.set_trace()
        setN = 1 #<-- Change number for this to get values from resp sets. 
        experiment1Tuples: list[tuple] = [(128,256,0.45), (64,128,0.45), (128,256,0.3), ("""NOTE: Other tuples can change one or more parameters whilst keeping at least one constant""")]
        batch_size = experiment1Tuples[setN][0] 
        hidden_units = experiment1Tuples[setN][1]
        dropout = experiment1Tuples[setN][2]
        
        print("---End of Experiment Setup #1---")
        return
    global listVerOfX_train
    listVerOfX_train = x_train.columns[:len(x_train.columns)-1].to_list()
    def experimentSetup2(numQuantLvls = 2):
        # Goal of exp: Want to see model performance based on degree of quantanization of data
        print("---Undergoing Experiment Setup #2---")
        # NOTE: Below will involve replacing 255 with a different number based on degree of quantanization of data. 
        # UPDATE: Only thing left is pulling maximum value from x_train and min value from train and test respectively. 
        # Using for loop to iterate through each column to apply this quantinization to each column. 
        global x_train, x_test
        print("---DEBUGGING CHECKPOINT #6: Ensuring that quantinization is doing at least something---")
        pb.set_trace()
        for i in listVerOfX_train:
            x_trainMax = x_train[i].max()
            x_trainMin = x_train[i].min()
            x_testMax = x_test[i].max()
            x_testMin = x_test[i].min()
            
            # widthsOfQuant = []
            # widthsOfQuant[0] = (x_trainMax - x_trainMin)/(numQuantLvls - 1),
            # widthsOfQuant[1] = (x_testMax - x_testMin)/(numQuantLvls - 1)
            widthsOfQuant = [(x_trainMax - x_trainMin)/(numQuantLvls - 1),(x_testMax - x_testMin)/(numQuantLvls - 1)]
            # x_train = np.reshape(x_train, [-1,input_size])
            # x_trainMax = 0;
            # x_trainMin = 0;
            
            # x_test = np.reshape(x_test, [-1,input_size])
            # x_testMax = 0;
            # x_testMin = 0;
            # widthsOfQuant[1] = (x_testMax - x_testMin)/(numQuantLvls - 1); 
            x_train.loc[:,i] = x_train.loc[:,i]/widthsOfQuant[0]
            x_test.loc[:,i] = x_test.loc[:,i]/widthsOfQuant[0]
            # x_test = x_test.astype('float32')//widthsOfQuant[1]
        print("---End of Experiment Setup #2---")
        return

    def experimentSetup3():
        # Goal of exp: Want to see model performance based on type of activation function from a subset of all possible activation functions. 
        print("---Undergoing Experiment Setup #3---" if isExperimentSetup3Active else "---Experiment Setup #3 was skipped---")
        activationFuncs = ['elu', 'sigmoid', 'tanh' ]
        # model.add(Activation(activationFuncs[0])) #<-- May need this since I have a classification problem that I'm wokring on. [UPDATE: Made this a return value instead] 

        print("---End of Experiment Setup #3---" if isExperimentSetup3Active else "---End of Experiment Setup #3 was skipped---")

        return model.add(Activation(activationFuncs[0])) if isExperimentSetup3Active == True else None #<-- May need this since I have a classification problem that I'm wokring on. 

    def experimentSetup4():
        print("---Undergoing Experiment Setup #4---")
        print("---End of Experiment Setup #4---")
        return


    # "Model is a 3-layer ML with ReLU and dropout after each layer": 
    #model = Sequential()
    #model.add(Dense(hidden_units,input_dim=input_size))
    #model.add(Activation('relu')) #<-- This is used to add activation function to model
    #model.add(Dropout(dropout))
    #model.add(Dense(hidden_units))
    #model.add(Activation('relu'))
    #model.add(Dropout(dropout))
    #model.add(Dense(num_labels))
    # "This is the output for one-hot vector"
    #model.add(Activation('softmax')) #<-- May need this since I have a classification problem that I'm wokring on. 
    #model.summary()
    #plot_model(model,to_file='mlp-mnist.png', show_shapes=True)

    # "Loss Function for one-hot vector"
    # "Use of adam optimizer"
    # "Accuracy is good metric for classification tasks": 
    # print("---DEBUGGING CHECKPOINT #2: Making attempt to test before running Experiment 0---")
    # pb.set_trace()#[experimentSetup0 was successful]
    
    experimentSetup0() 
    # experimentSetup1() 
    experimentSetup2() 
    # "This is the output for one-hot vector"
    model.add(Activation('softmax')) #<-- May need this since I have a classification problem that I'm wokring on. 
    model.summary()
    #plot_model(model,to_file='mlp-mnist.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    # "Train the network"
    # BUG: There is a bug that occurs here that prods the following message: ValueError: Unrecognized data type: x=  [Apparently, there is an unrecognized data type in x_train variable]
    x_train.loc[:,"Company"] = pd.Series([2**i for i in range(x_train.shape[0])]) 
    x_test.loc[:,"Company"] = pd.Series([2**i for i in range(x_train.shape[0])]) 
    x_train = x_train.astype("float64")
    x_test = x_test.astype("float64")
    x_train.loc[:,"Company"] = x_train.loc[:,"Company"].astype("int32")
    x_test.loc[:,"Company"] = x_test.loc[:,"Company"].astype("int32")
    # y_train = pd.Series(y_train)
    # y_test = pd.Series(y_test)
    # NOTE: Body of Problem is coming from x_train and x_test!
    x_trainCopy = tf.data.Dataset.from_tensor_slices((x_train.values.astype(np.float32),tf.convert_to_tensor(y_train).numpy().astype(np.float32)))
    x_testCopy = tf.data.Dataset.from_tensor_slices((x_test.values.astype(np.float32),tf.convert_to_tensor(y_test).numpy().astype(np.float32)))
    # NOTE: Body of Problem is coming from x_train and x_test!
    # x_train = x_trainCopy 
    # x_test = x_testCopy 

    print("---DEBUGGING CHECKPOINT #3: Making attempt to test before training network---")
    pb.set_trace()
    # UPDATE: FINISHED MODEL!!
    train_history = model.fit(x_train,y_train,epochs=20, batch_size=batch_size) #<-- Removing this since I converted dataframe into tensorflow version of dataframe. [UPDATE: Have a problem now coming from mismatching shapes for x_train and y_train. Error is as follows: "ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 1), output.shape=(None, 4)"]
    # train_history = model.fit(x_trainCopy,epochs=20, batch_size=batch_size) #<-- Removing this since I converted dataframe into tensorflow version of dataframe. 
    acc = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    # acc = model.evaluate(x_testCopy)
    # print("---DEBUGGING CHECKPOINT #4: Obtaining Test Accuracy---")
    # pb.set_trace() [COMPLETE!]
    # print("\nTest accuracy: %.1f%%" % (100.0 * acc))
    print("\nTest accuracy: %.1f%%" % (100.0 * acc[1]))
    test_stocks["Company"] = test_stocks.index; test_stocks.loc[:,"Company"] = test_stocks.loc[:,"Company"].astype("int32")

    # 6) Verifying and Visualzing the Predictions
    # a) Obtaining accuarrcy of the predictions: 
    
    # predictions = model.predict(test_stocks) #<-- NOTE: This retunrns an array of probabilities for each class. Thus, for future consumption by person, could assign these predictions to a column added to test_stocks?
    predictions = model.predict(x_test) #<-- NOTE: This retunrns an array of probabilities for each class. Thus, for future consumption by person, could assign these predictions to a column added to test_stocks?
    
    predictions[0]
    
    
    np.argmax(predictions[0]) #<-- Returns the prediction that Neural Network was most comfortable with. 
    # def plot_image(i, predictions_array, true_label, img):
        # true_label, img = true_label[i], img[i]
        # plt.grid(False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(img, cmap=plt.cm.binary)
        # predicted_label = np.argmax(predictions_array)

        # if predicted_label == true_label:
        #     color = 'blue'
        # else:
        #     color = 'blue'
        
        # plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label], color=color))
                   # NOTE as of 10/28/25: plot_image function will NOT be needed
    # Body of original idea for plotting model's output: 
    def plot_value_array(i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#7777777")
        plt.ylim([0,1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
        return

    # Below can potentially be done iteratively?
    """ 
    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    # plot_image(i,predictions[i],test_labels,test_images) <-- Not needed. 
    plt.subplot(1,2,2)
    plot_value_array(i,predictions[i],test_labels)
    plt.show()
    """
    # End of Body of original idea for plotting model's output: [UPDATE: Commented out this stuff above]


    # Alternate way of doing plotting above[for clarity, alternative is referenced BELOW]. 
    print("---DEBUGGING CHECKPOINT #5: Tesiting plotting!---")
    pb.set_trace() # [COMPLETE!]
    # Body of plotting model[Link for plotting is here: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/] 

    ## Listing all data in history: 
    print(train_history.history.keys())

    ## summarize train_history for accuracy
    plt.plot(train_history.history['accuracy'])
    # print(train_history.history['val_accuracy']) #<-- NOTE: val_accuracy and val_loss are NOT part of the history object's keys.  
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    ## summarize history for loss
    plt.plot(train_history.history['loss'])
    # print(train_history.history['val_loss']) #<-- NOTE: val_accuracy and val_loss are NOT part of the history object's keys. 
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    
    # End of Body of plotting model 
    # b) Printing Model Summary, which can be good to go into detail about: 
    model.summary()





    # end of 6)
# """
attempt3()
