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
    dataset = dataset.shuffle(buffer_size=x.shape[0].batch(batch)size))

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


def attempt3():
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.utils import to_categorical, plot_model
    from tensorflow.keras.datasets import mnist

    # "load mnist dataset"
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    filepathToTrainingSet: str
    filePathToTrainingSet += f"/trainingSetStrat{1}.py" #<-- number will change based on strat being used.
    filepathToTestSet: str
    filePathToTrainingSet += f"/testSetStrat{1}.py" #<-- number will change based on strat being used.
    train_stocks = pd.read_csv(f"{filepathToTrainingSet}"); train_labels = test_labels = ["will reference company names"];
    test_stocks = pd.read_csv(f"{filepathToTestSet}"); test_labels = ["will reference company names"];
    x_train = train_stocks.loc[:,df.columns != "Optimality"]
    y_train = train_labels.loc[:,"Optimality"]
    x_test = test_stocks.loc[:,df.columns != "Optimality"]
    y_test = test_labels.loc[:,df.columns != "Optimality"]
    # ^^ Above ensures that prediction labels are y and x references the data used to make said decision. 
            

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
    batch_size = 128
    hidden_units = 256
    dropout = 0.45

    # "Model is a 3-layer ML with ReLU and dropout after each layer": 
    model = Sequential()
    model.add(Dense(hidden_units,input_dim=input_size))
    model.add(Activation('relu')) #<-- This is used to add activation function to model
    model.add(Dropout(dropout))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_labels))
    # "This is the output for one-hot vector"
    model.add(Activation('softmax')) #<-- May need this since I have a classification problem that I'm wokring on. 
    model.summary()
    plot_model(model,to_file='mlp-mnist.png', show_shapes=True)

    # "Loss Function for one-hot vector"
    # "Use of adam optimizer"
    # "Accuracy is good metric for classification tasks": 
    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    # "Train the network"
    model.fit(x_train,y_train,epochs=20, batch_size=batch_size)
    acc = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))

    # 6) Verifying and Visualzing the Predictions
    # a) Obtaining accuarrcy of the predictions: 
    
    predictions = probability_model.predict(test_stocks)
    
    predictions[0]
    
    
    np.argmax(predictions[0]) #<-- Returns the prediction that Neural Network was most comfortable with. 
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

               # Below can potentially be done iteratively?
    i = 0
    plt.figure(figsize(6,3))
    plt.subplot(1,2,1)
    plot_image(i,predictions[i],test_labels,test_images)
    plt.subplot(1,2,2)
    plot_image(i,predictions[i],test_labels)
    plt.show()

    # b) Printing Model Summary, which can be good to go into detail about: 
    model.summary()

    # end of 6)
attempt3()
