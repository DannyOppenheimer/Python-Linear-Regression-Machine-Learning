# Python Linear Regression Machine Learning using Tensorflow Libraries
# ---Credit---
# Paulo Cortez, University of Minho, GuimarÃ£es, Portugal, http://www3.dsi.uminho.pt/pcortez
# UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/student+performance]

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import style
import pickle

# initial options for user
train_AI = False
want_train = input("Would you like to train the AI? [y/n]: ")
if want_train.lower() == "y" or want_train.lower() == "yes" or want_train.lower() == "yeah":
    train_AI = True
    times_train = input("How many training rounds? (don't break computer pl0x)? [num]: ")
else:
    train_AI = False

# read .csv file
student_data = pd.read_csv("student-dat.csv", sep=";")
student_data = student_data[["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrelquality", "freetime", "goout", "dayalcohol", "weekendalcohol", "health", "absences", "G1", "G2", "G3"]]

# we want to predict the end grade, so the prediction goal is G3
goalPrediction = "G3"

# X is the set without the final grade (G3) to find all the values we want to train off of
# Y is the final grad (G3) that we want to predict
X = np.array(student_data.drop(goalPrediction, 1))
Y = np.array(student_data[goalPrediction])

# x_train and y_train is training data the computer can read and learns off of
# x_test and y_test is the prediction data the computer is given to calculate new final grade
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


# best current data set, saved in bestLearn.txt
listBest = open("bestLearn.txt", "r").readlines(1)
winningDatSet = float(listBest[0].translate({ord('b'): None}))

# has the training rounds yielded any better data sets?
foundMoreAccDat = False

# training section
if train_AI:
    for x in range(int(times_train)):
        # refresh the test pool with every run through
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

        # fit a linear regression like to the data collected
        linear_regress = linear_model.LinearRegression()
        linear_regress.fit(x_train, y_train)

        # accuracy of best fit line
        linear_acc = linear_regress.score(x_test, y_test)
        print("Training round ", x + 1, " had an accuracy of ", "%.01f" % (linear_acc * 100), "%", sep="")

        # did any of the training rounds surpass the pickled data file's accuracy?
        if linear_acc > winningDatSet:
            foundMoreAccDat = True
            winningDatSet = linear_acc

            # save that value to a txt file
            with open("Test Scores/bestLearn.txt", "w") as h:
                h.write(str(winningDatSet))
            print("Training round ", x + 1, " has the running best accuracy of ", "%.01f" % (winningDatSet * 100), "%", sep="")

            # create a new pickle file if there isn't one, or write to an existing one
            with open("studentModel.pickle", "wb") as f:
                # dump linear regression into above pickle file
                pickle.dump(linear_regress, f)

    if foundMoreAccDat:
        print("Final accuracy: ", "%.01f" % (winningDatSet * 100), "%", sep="")
    else:
        print("This training run didn't get a higher accuracy than a saved one :( current set has an accuracy of: ", "%.01f" % (winningDatSet * 100), "%", sep="")


# open pickle file to access other information
pickleIn = open("studentModel.pickle", "rb")
linear_regress_data = pickle.load(pickleIn)
print("\n***********************************************")
print("* Coefficients: ", linear_regress_data.coef_)
print("* Intercept: ", linear_regress_data.intercept_)
print("***********************************************")

# print out computer's predictions for the x_test data
print("\nHow to read: <guessed score> [students background] <actual score>")
predictions = linear_regress_data.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# graph plotting to see the relationships for individual pieces of data
def ask_graph():
    graph_req = input("\nWhich graph would you like to display?")
    try:
        style.use("ggplot")
        pyplot.scatter(student_data[graph_req], student_data["G3"])
        pyplot.xlabel(graph_req)
        pyplot.ylabel("Final Grade")
        pyplot.show()
    except KeyError:
        print("Oops! That's not an option. These are though!")
        print("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrelquality", "freetime", "goout", "dayalcohol", "weekendalcohol", "health", "absences", "G1", "G2", "G3", sep=", ")
        ask_graph()


# prompt graph
exit_request = input("Would you like to see a graph? [y/n]: ")

if exit_request.lower() == "y" or exit_request.lower() == "yes" or exit_request.lower() == "yeah":
    ask_graph()

# User indicates whether or not they would like to save the most accurate data set
overwrite_learn = input("Would you like to overwrite the most accurate data set and set it to 0? [y/n]: ")
if overwrite_learn.lower() == "y" or overwrite_learn.lower() == "yes" or overwrite_learn.lower() == "yeah":
    with open("Test Scores/bestLearn.txt", "w") as h:
        h.write(str("0.0"))
