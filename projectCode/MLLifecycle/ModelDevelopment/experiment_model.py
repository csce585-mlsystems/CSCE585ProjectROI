# experiment_model.py
# variant of the model with experiment hooks turned on (quantization, etc.)

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


def ModelTrainingAndDevelopment():
    """
    Same idea as attempt3, but:
    - explicitly calls experimentSetup2 (quantization)
    - otherwise keeps the same structure
    """
    inNoteBook = False

    filePathToModelDir = (
        "C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/"
        "ProjectRepo/projectCode/MLLifecycle/ModelDevelopment/preparedDataset.csv"
        if inNoteBook is False
        else "preparedDataset.csv"
    )

    # load data
    train_stocks = pd.read_csv(filePathToModelDir)
    train_labels = train_stocks.loc[:, "Company"]
    test_stocks = pd.read_csv(filePathToModelDir)
    test_labels = train_stocks.loc[:, "Company"]

    del train_stocks["Unnamed: 0"]
    del test_stocks["Unnamed: 0"]

    # features + labels
    global x_train, x_test
    x_train = train_stocks.loc[:, train_stocks.columns != "Optimality"]
    y_train = train_labels
    x_test = test_stocks.loc[:, train_stocks.columns != "Optimality"]
    y_test = test_labels

    num_labels = len(np.unique(y_train))

    # one-hot labels
    y_train = to_categorical(y_train.index)
    y_test = to_categorical(y_test.index)

    input_size = len(train_stocks.columns)

    # network hyper-params
    global batch_size, hidden_units, dropout
    global model
    batch_size = 128
    hidden_units = 256
    dropout = 0.45
    isExperimentSetup3Active = False

    model = Sequential()

    # experiment helpers

    def experimentSetup0():
        """
        default network structure for this experiment version.
        """
        print("---Running Experiment Setup #0 (experiment version)---")
        model.add(Input((x_train.shape[1],)))
        model.add(Dense(hidden_units))
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_units))
        model.add(Activation("relu") if experimentSetup3() is None else experimentSetup3())
        model.add(Dropout(dropout))
        model.add(Dense(num_labels))
        print("---End of Experiment Setup #0---")

    def experimentSetup1():
        """
        hyper-param tuning helper (not used by default).
        """
        print("---Running Experiment Setup #1 (hyperparam variations)---")
        setN = 1
        experiment1Tuples: list[tuple] = [
            (128, 256, 0.45),
            (64, 128, 0.45),
            (128, 256, 0.3),
            ("NOTE: placeholder for more tuples"),
        ]
        batch_size = experiment1Tuples[setN][0]
        hidden_units = experiment1Tuples[setN][1]
        dropout = experiment1Tuples[setN][2]
        print("---End of Experiment Setup #1---")

    global listVerOfX_train
    listVerOfX_train = x_train.columns[: len(x_train.columns) - 1].to_list()

    def experimentSetup2(numQuantLvls=2):
        """
        apply basic quantization to each feature column.
        this version actually gets called in this function.
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
        optional activation-swap hook.
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
        placeholder for future experiments.
        """
        print("---Running Experiment Setup #4 (placeholder)---")
        print("---End of Experiment Setup #4---")

    #  build + train model 

    experimentSetup0()
    # experimentSetup1()
    experimentSetup2()  # this version actually uses the quantization experiment

    model.add(Activation("softmax"))

    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    # same company-ID trick as the baseline
    x_train.loc[:, "Company"] = pd.Series([2**i for i in range(x_train.shape[0])])
    x_test.loc[:, "Company"] = pd.Series([2**i for i in range(x_train.shape[0])])

    x_train = x_train.astype("float64")
    x_test = x_test.astype("float64")

    x_train.loc[:, "Company"] = x_train.loc[:, "Company"].astype("int32")
    x_test.loc[:, "Company"] = x_test.loc[:, "Company"].astype("int32")

    x_trainCopy = tf.data.Dataset.from_tensor_slices(
        (x_train.values.astype(np.float32), tf.convert_to_tensor(y_train).numpy().astype(np.float32))
    )
    x_testCopy = tf.data.Dataset.from_tensor_slices(
        (x_test.values.astype(np.float32), tf.convert_to_tensor(y_test).numpy().astype(np.float32))
    )

    train_history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

    acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc[1]))

    test_stocks["Company"] = test_stocks.index
    test_stocks.loc[:, "Company"] = test_stocks.loc[:, "Company"].astype("int32")

    predictions = model.predict(x_test)
    predictions[0]

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

    model.summary()


if __name__ == "__main__":
    ModelTrainingAndDevelopment()
