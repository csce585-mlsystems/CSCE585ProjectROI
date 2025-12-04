# baseline_model.py
# basic baseline model training using the preparedDataset.csv file

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

def attempt3():
    """
    Baseline model:
    - load preparedDataset.csv
    - build a dense network
    - train it and write predictions to a CSV
    """
    inNoteBook = True  # <- run in local repo folder so we use preparedDataset.csv

    # path to the CSV that has the engineered stock features
    filePathToModelDir = (
        "C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/"
        "ProjectRepo/projectCode/MLLifecycle/ModelDevelopment/preparedDataset.csv"
        if inNoteBook is False
        else "preparedDataset.csv"
    )

    # read the data once for train + test (they’re the same file here)
    train_stocks = pd.read_csv(filePathToModelDir)
    train_labels = train_stocks.loc[:, "Company"]
    test_stocks = pd.read_csv(filePathToModelDir)
    test_labels = train_stocks.loc[:, "Company"]

    # drop the auto index column that got written to the CSV
    del train_stocks["Unnamed: 0"]
    del test_stocks["Unnamed: 0"]

    # x = features, y = labels (company)
    global x_train, x_test
    x_train = train_stocks.loc[:, train_stocks.columns != "Optimality"]
    y_train = train_labels
    x_test = test_stocks.loc[:, train_stocks.columns != "Optimality"]
    y_test = test_labels

    # number of unique labels (companies)
    num_labels = len(np.unique(y_train))

    # keras expects one-hot labels
    y_train = to_categorical(y_train.index)
    y_test = to_categorical(y_test.index)

    input_size = len(train_stocks.columns)  # number of input features per row

    # network hyper-params
    global batch_size, hidden_units, dropout
    global model
    batch_size = 128
    hidden_units = 256
    dropout = 0.45
    isExperimentSetup3Active = False

    model = Sequential()

    # experiment setup helpers

    def experimentSetup0():
        """
        default model architecture:
        Input -> Dense -> ReLU -> Dropout -> Dense -> ReLU (maybe changed)
        -> Dropout -> Dense(num_labels)
        """
        print("---Running Experiment Setup #0 (baseline)---")
        model.add(Input((x_train.shape[1],)))
        model.add(Dense(hidden_units))
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units))
        # activation here can be swapped by experimentSetup3
        model.add(Activation("relu") if experimentSetup3() is None else experimentSetup3())
        model.add(Dropout(dropout))
        model.add(Dense(num_labels))
        print("---End of Experiment Setup #0---")

    def experimentSetup1():
        """
        toy helper for changing batch size / hidden units / dropout.
        not being used right now, but keeping it for reference.
        """
        print("---Running Experiment Setup #1 (hyperparam variations)---")
        setN = 1  # choose which tuple to use
        experiment1Tuples: list[tuple] = [
            (128, 256, 0.45),
            (64, 128, 0.45),
            (128, 256, 0.3),
            ("NOTE: placeholder for more tuples"),
        ]
        # these are local reassignments; original globals stay as written above
        batch_size = experiment1Tuples[setN][0]
        hidden_units = experiment1Tuples[setN][1]
        dropout = experiment1Tuples[setN][2]
        print("---End of Experiment Setup #1---")

    # list of feature columns (except the last one)
    global listVerOfX_train
    listVerOfX_train = x_train.columns[: len(x_train.columns) - 1].to_list()

    def experimentSetup2(numQuantLvls=2):
        """
        quick quantization helper for features (not used here by default).
        scales columns into a fixed number of quantization levels.
        """
        print("---Running Experiment Setup #2 (quantization)---")
        global x_train, x_test
        for col in listVerOfX_train:
            x_trainMax = x_train[col].max()
            x_trainMin = x_train[col].min()
            x_testMax = x_test[col].max()
            x_testMin = x_test[col].min()

            widthsOfQuant = [
                (x_trainMax - x_trainMin) / (numQuantLvls - 1),
                (x_testMax - x_testMin) / (numQuantLvls - 1),
            ]

            x_train.loc[:, col] = x_train.loc[:, col] / widthsOfQuant[0]
            x_test.loc[:, col] = x_test.loc[:, col] / widthsOfQuant[0]

        print("---End of Experiment Setup #2---")

    def experimentSetup3():
        """
        optional hook for swapping the activation function on the hidden layer.
        """
        print(
            "---Running Experiment Setup #3---"
            if isExperimentSetup3Active
            else "---Experiment Setup #3 is skipped---"
        )
        activationFuncs = ["elu", "sigmoid", "tanh"]
        print(
            "---End of Experiment Setup #3---"
            if isExperimentSetup3Active
            else "---End of Experiment Setup #3 (skipped)---"
        )
        return model.add(Activation(activationFuncs[0])) if isExperimentSetup3Active else None

    def experimentSetup4():
        """
        placeholder for any future experiments (e.g., scaling analysis).
        """
        print("---Running Experiment Setup #4 (placeholder)---")
        print("---End of Experiment Setup #4---")

    # build + train model

    experimentSetup0()
    # experimentSetup1()
    # experimentSetup2()

    # final output activation for multi-class classification
    model.add(Activation("softmax"))

    model.summary()

    # compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    # convert "Company" to integers and use as one of the features
    x_train.loc[:, "Company"] = pd.Series([2**i for i in range(x_train.shape[0])])
    x_test.loc[:, "Company"] = pd.Series([2**i for i in range(x_train.shape[0])])

    x_train = x_train.astype("float64")
    x_test = x_test.astype("float64")

    x_train.loc[:, "Company"] = x_train.loc[:, "Company"].astype("int32")
    x_test.loc[:, "Company"] = x_test.loc[:, "Company"].astype("int32")

    # keep tf dataset versions around if needed later
    x_trainCopy = tf.data.Dataset.from_tensor_slices(
        (x_train.values.astype(np.float32), tf.convert_to_tensor(y_train).numpy().astype(np.float32))
    )
    x_testCopy = tf.data.Dataset.from_tensor_slices(
        (x_test.values.astype(np.float32), tf.convert_to_tensor(y_test).numpy().astype(np.float32))
    )

    # actually train
    train_history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

    # eval on "test" data (which is just the same CSV)
    acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc[1]))

    # keep a copy of test_stocks for later / debugging
    test_stocksCopy = test_stocks

    # add integer company IDs to test df
    test_stocks["Company"] = test_stocks.index
    test_stocks.loc[:, "Company"] = test_stocks.loc[:, "Company"].astype("int32")

    # get predictions from the trained model
    WriteModelToAFile = True
    predictions = model.predict(x_test)

    # if this flag is on, write predictions out to a CSV
    local = True  # this controls the path that gets used
    if WriteModelToAFile:
        print("---Writing Model predictions to CSV for future use---")
        print("---DEBUG CHECKPOINT: entering pdb before write---")
        pb.set_trace()

        copyTwoSendOff = test_stocks.copy()
        # keep only the predicted class index for each row
        copyTwoSendOff["Model Predictions"] = pd.Series([np.argmax(x) for x in predictions])

        # convert company labels back to original names
        copyTwoSendOff.loc[:, "Company"] = test_labels

        writePathForModelPreds = (
            "ModelPredictions.csv"  # write next to this script instead of Windows path
            if local
            else ""
        )

        copyTwoSendOff.to_csv(writePathForModelPreds)
        print("---Done writing model predictions CSV---")

    predictions[0]

    # simple accuracy / loss plots
    print(train_history.history.keys())

    plt.plot(train_history.history["accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(train_history.history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    # print model summary again at the end
    model.summary()


if __name__ == "__main__":
    attempt3()
