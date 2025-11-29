# Purpose: This file will contain code that helps with Model Development. 

# Body of neccessary imports
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np #<-- May be optional not sure as of 10/13/25. 
# import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
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
    filePathToTrainingSetDir: str = "./ModelDevelopment/TrainingSets" #<-- fill in later[complete]
    filepathToTrainingSet: str = filePathToTrainingSetDir
    filePathToTrainingSet += f"/trainingSetStrat{1}.py" #<-- number will change based on strat being used.
    filePathToTestSetDir: str = "MLLifecycle/ModelDevelopment/TestingSets" #<-- fill in later[complete]
    filepathToTestSet: str = filePathToTestSetDir
    filePathToTestSet += f"/testSetStrat{1}.py" #<-- number will change based on strat being used.
    train_stocks = pd.read_csv(f"{filepathToTrainingSet}"); train_labels = test_labels = ["will reference company names"] or train_stocks.loc["company"];#<-- References labels which are derived from custom engineered dataset. 
    test_stocks = pd.read_csv(f"{filepathToTestSet}"); test_labels = train_stocks.loc["company"] or ["will reference company names"]; #<-- References labels which are derived from custom engineered dataset. 
    x_train = train_stocks.loc[:,train_stocks.columns != "Optimality"]
    y_train = train_labels.loc[:,"Optimality"]
    x_test = test_stocks.loc[:,train_stocks.columns != "Optimality"]
    y_test = test_labels.loc[:,train_stocks.columns != "Optimality"]
    # ^^ Above ensures that prediction labels are y and x references the data used to make said decision. [UPDATE: Due to this, will need to make a change in data prep folder]
            

    # "Compute the number of labels"
    num_labels = len(np.unique(y_train))

    # "Convert to one-hot vector"[we converted the labels to one-hot vectors using to_categorical]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # ----May snippet below may be optional?---
    # "image dimensions(assumed square)"
    image_size = x_train.shape[1]
    input_size = image_size * image_size

    # "Resize and Normalize"
    x_train = np.reshape(x_train, [-1,input_size])
    x_train = x_train.astype('float32')/255
    x_test = np.reshape(x_test, [-1,input_size])
    x_test = x_test.astype('float32')/255

    # ----May snippet above may be optional?[UPDATE: Above is required, need to come up with normalization process]---
    # "Network Parameters"
    # NOTE: The following will reference a list of tuple(s) that will be used to facilitate first experiment. 
    listOfExperimentSetupFuncs = ["""Plan to insert functions for doing experiment(s) setup here"""]
    global batch_size, hidden_units, dropout
    batch_size = 128
    # batch_size = experiment1Tuples[0][0] 
    hidden_units = 256
    # hidden_units = experiment1Tuples[0][1]
    dropout = 0.45
    # dropout  = experiment1Tuples[0][2]
    isExperimentSetup3Active = False
    def experimentSetup0():
       # This will reference the default settings irrespective to experiments. 
        print("---Undergoing Experiment Setup #0---")
        model = Sequential()
        model.add(Dense(hidden_units,input_dim=input_size))
        model.add(Activation('relu')) #<-- This is used to add activation function to model[UPDATE: May need to replace Activation('relu') by making them default...not sure to facilitate experimentSetup3 OR I can simply see what happens when the 2nd to LAST acivation is changed] 
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units))
        model.add(Activation('relu')) if experimentSetup3() == None else experimentSetup3 #<-- This changes the Activation function, adhering to experimentSetup0![Thus, as of 11/23/25: There are two experiments ready to go for Milestone 1]
        model.add(Dropout(dropout))
        model.add(Dense(num_labels))
        print("---End of Experiment Setup #0---")
        return

    def experimentSetup1():
        # Goal of exp: Want to see how model params affect the acuaracy of the model by modifying batch size and hidden units and dropout[pn: make sure to have a subset of the powerset of params be modified in this exp. ]
        print("---Undergoing Experiment Setup #1---")
        experiment1Tuples: list[tuple] = [(128,256,0.45), (64,128,0.45), ("""NOTE: Other tuples can change one or more parameters whilst keeping at least one constant""")]
        batch_size = experiment1Tuples[0][0] 
        hidden_units = experiment1Tuples[0][1]
        dropout = experiment1Tuples[0][2]
        
        print("---End of Experiment Setup #1---")
        return
    def experimentSetup2(numQuantLvls = 2):
        # Goal of exp: Want to see model performance based on degree of quantanization of data
        print("---Undergoing Experiment Setup #2---")
        # NOTE: Below will involve replacing 255 with a different number based on degree of quantanization of data. 
        # UPDATE: Only thing left is pulling maximum value from x_train and min value from train and test respectively. 
        widthsOfQuant = []
        
        widthsOfQuant[0] = (x_trainMax - x_trainMin)/(numQuantLvls - 1)
        x_train = np.reshape(x_train, [-1,input_size])
        x_trainMax = 0;
        x_trainMin = 0;
        
        x_test = np.reshape(x_test, [-1,input_size])
        x_testMax = 0;
        x_testMin = 0;
        widthsOfQuant[1] = (x_testMax - x_testMin)/(numQuantLvls - 1); 
        x_train = x_train.astype('float32')/widthsOfQuant[0]
        x_test = x_test.astype('float32')//widthsOfQuant[1]
        print("---End of Experiment Setup #2---")
        return

    def experimentSetup3():
        # Goal of exp: Want to see model performance based on type of activation function from a subset of all possible activation functions. 
        print("---Undergoing Experiment Setup #3---")
        activationFuncs = ['elu', 'sigmoid', 'tanh' ]
        # model.add(Activation(activationFuncs[0])) #<-- May need this since I have a classification problem that I'm wokring on. [UPDATE: Made this a return value instead] 

        print("---End of Experiment Setup #3---")
        return model.add(Activation(activationFuncs[0])) if isExperimentSetup3Active == True else None #<-- May need this since I have a classification problem that I'm wokring on. 

    def experimentSetup4():
        print("---Undergoing Experiment Setup #4---")
        print("---End of Experiment Setup #4---")
        return


    # "Model is a 3-layer ML with ReLU and dropout after each layer": 
    global model #<-- Have this here, so experimental setup functions can tweak model as needed. 
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
    
    experimentSetup0() 
    # "This is the output for one-hot vector"
    model.add(Activation('softmax')) #<-- May need this since I have a classification problem that I'm wokring on. 
    model.summary()
    #plot_model(model,to_file='mlp-mnist.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    # "Train the network"
    train_history = model.fit(x_train,y_train,epochs=20, batch_size=batch_size)
    acc = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))

    # 6) Verifying and Visualzing the Predictions
    # a) Obtaining accuarrcy of the predictions: 
    
    predictions = model.predict(test_stocks) #<-- NOTE: This retunrns an array of probabilities for each class. Thus, for future consumption by person, could assign these predictions to a column added to test_stocks?
    
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
    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    # plot_image(i,predictions[i],test_labels,test_images) <-- Not needed. 
    plt.subplot(1,2,2)
    plot_value_array(i,predictions[i],test_labels)
    plt.show()
    # End of Body of original idea for plotting model's output: 


    # Alternate way of doing plotting above[for clarity, alternative is referenced BELOW]. 
    # Body of plotting model[Link for plotting is here: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/] 

    ## Listing all data in history: 
    print(train_history.train_history.keys())

    ## summarize train_history for accuracy
    print(train_history.train_history['accuracy'])
    print(train_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuarcy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    ## summarize history for loss
    print(train_history.train_history['loss'])
    print(train_history.train_history['val_loss'])
    plt.title('model loss')
    plt.ylabel('accuarcy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    
    # End of Body of plotting model 
    # b) Printing Model Summary, which can be good to go into detail about: 
    model.summary()





    # end of 6)
# """
attempt3()
