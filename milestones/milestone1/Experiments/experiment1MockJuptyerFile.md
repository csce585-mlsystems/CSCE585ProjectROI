NOTE: Got the Alpah Vatnage API Key. This will help with getting the remaining independent vars for strat 1. THe key is: 7ZF36PV80X3ELGX9. 
[Purpose]: <> "This file will be a mock file that'll serve as a guide for mapping to .ipynb file of same name in current directory"
[Experiments I plan to employ]: <> "Want to see how well it holds up compared to benchmark data set, Want to see how the model parameters affect the accuracy of the model, can do a test involving using different activation functions, model performance based on degree of quantanization of data when making decisions[with focus on time it takes for model to produce an answer], [could do something involving time for computation?]"

## Experiment 1[<Insert title of Experiment>]
### Purpose
- Setting up Classification problem and visualizing the model accuracy based on permutations of 3-tuple that references the model parameters. 
### Hypothesis
- I believe that the ratio of constants in the 3-tuple should be <> of the ML Model paramters for the ML Model's accurarcy to be greater than the ML Model's accuracy when the ratio is not used. 


### Instructions to Run
Instructions: 1) Setting up Model, 2) Using three situations where the 3-tupeles are different.
- Context: First we run code that creates the datasets needed to be fed to the model: [note, make habit to employ content-codeSnippet pairs!]
```py
    # Fill in this portion with data engineering procedure
```
- Context: First we do the data preprocessing before it meets with the model[part a)]: [note, make habit to employ content-codeSnippet pairs!]
```py
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
```
- Context: Then we set up the model parameters[part b)]: [note, make habit to employ content-codeSnippet pairs!]
```py
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
    #plot_model(model,to_file='mlp-mnist.png', show_shapes=True)

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
    
    predictions = model.predict(test_stocks)
    
    predictions[0]
    
    
    np.argmax(predictions[0]) #<-- Returns the prediction that Neural Network was most comfortable with. 
```
### Results 
- Context: [note, make habit to employ content-codeSnippet pairs![Make sure results section consists of python code that utilizes data visualization]
```py
# Insert code that displays data visualization here
```
### Conclusion
- Context: [make sure this portion consists of summary of results and determine if hypothesis was met or unmet]
### [cont here with subheadings IF applicable. Rfer to this link for reference: https://github.com/csce585-mlsystems/instructions/blob/main/instructions_project_rubric.md][UPDATE: Don't think any others are needed]
[NOTE: Experiments succeeding this point will be implemented in future]
## Experiment 2[<Insert title of Experiment>][Could involve comparing performance between MLP and Convolutional Neural Network on CPU?]
### Purpose

### Hypothesis


### Instructions to Run
- Context: [note, make habit to employ content-codeSnippet pairs!]
```py
```
### Results 
- Context: [note, make habit to employ content-codeSnippet pairs!]
```py
```
### Conclusion
- Context: [make sure this portion consists of summary of results and determine if hypothesis was met or unmet]
### [cont here with subheadings IF applicable. Rfer to this link for reference: https://github.com/csce585-mlsystems/instructions/blob/main/instructions_project_rubric.md][UPDATE: Don't think any others are needed]
## Experiment 3[<Insert title of Experiment>]
### Purpose

### Hypothesis


### Instructions to Run
- Context: [note, make habit to employ content-codeSnippet pairs!]
```py
```
### Results 
- Context: [note, make habit to employ content-codeSnippet pairs!]
```py
```
### Conclusion
- Context: [make sure this portion consists of summary of results and determine if hypothesis was met or unmet]
### [cont here with subheadings IF applicable. Rfer to this link for reference: https://github.com/csce585-mlsystems/instructions/blob/main/instructions_project_rubric.md][UPDATE: Don't think any others are needed]

