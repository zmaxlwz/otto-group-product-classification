import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

from sklearn.base import clone
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

#indicator specifying whether to use real test set or set split from training set
get_test=False
#set random seede
np.random.seed(11)

def get_data():
    global get_test
    #read training data
    train_data = np.loadtxt("../data/train.csv", dtype='string', delimiter=',', skiprows=1)
    #the first column is id, get features from the second column to the second last column
    train_features = train_data[:, 1:-1].astype(int)
    #the last column is the target variable
    train_target = train_data[:, -1]

    if get_test==True:
        #read test data
        test_data = np.loadtxt("../data/test.csv", dtype='int', delimiter=',', skiprows=1)
        #the first column is id, get features from the second column to the last column
        test_features = test_data[:, 1:]
    else:
        #don't read test data
        test_features = []

    return (train_features, train_target, test_features)

def print_class_ratio(y):
    #y is an array of class labels in the dataset
    #print class proportion in training set according to y
    for i in range(9):
        #print sum(y=='Class_' + str(i+1))
        #print sum(y=='Class_' + str(i+1))*1.0/len(y)
        print "Class_" + str(i+1) + " ratio: {0:.4f}" .format(sum(y=='Class_' + str(i+1))*1.0/len(y))

    # Class_1 ratio: 0.0312
    # Class_2 ratio: 0.2605
    # Class_3 ratio: 0.1294
    # Class_4 ratio: 0.0435
    # Class_5 ratio: 0.0443
    # Class_6 ratio: 0.2284
    # Class_7 ratio: 0.0459
    # Class_8 ratio: 0.1368
    # Class_9 ratio: 0.0801

def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll

def transform_output(y):
    #y is a array with Class_1 ~ Class_9
    #transform it into 0-1 integer array

    #length of y
    y_length = len(y)

    #category vector
    class_vec = np.array(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])

    #generate identity matrix
    m = np.identity(len(class_vec))

    output_matrix = class_vec
    #print output_matrix

    for i in np.arange(y_length):
        print i
        new_vec = m[np.where(y[i]==class_vec)[0], :].astype(int)
        #print new_vec
        output_matrix = np.vstack((output_matrix,new_vec))

    return output_matrix[1:,:]

def output_result(y_matrix):
    print "Output result..."

    y_temp = np.array(["%.6f" % x for x in y_matrix.reshape(y_matrix.size)])
    y_matrix = y_temp.reshape(y_matrix.shape)

    #print y_matrix[:10,:]

    data_length = y_matrix.shape[0]
    #generate id
    id_test = np.arange(1, data_length+1).astype(str)
    #insert column name 'id' to the beginning
    id_test = np.insert(id_test, 0, 'id')

    #category vector
    class_vec = np.array(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])

    y_matrix = np.vstack((class_vec, y_matrix))

    result = np.vstack((id_test,y_matrix.T)).T
    #print result[:10, :]

    np.savetxt("../output/result.csv", result, fmt='%s', delimiter=',', header='', footer='', comments='')

def plot_hist(data, bin_num, range_min, range_max):
    #plot the data histogram
    plt.hist(data, bins=bin_num, range=(range_min, range_max), color='b')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Single feature value histogram')
    #plt.xlim(range_min, range_max)
    #plt.ylim(0, 100)
    plt.show()


def uniform_prob_benchmark(x_test, y_test):
    global get_test

    #the number of test case
    test_cast_num = x_test.shape[0]
    #each tuple has 9 elements, which is the same, and sum to 1
    y_prob = np.ones((test_cast_num, 9))/9

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))




def randomForest(x_train, y_train, x_test, y_test):
    global get_test
    '''
    #use LDA to do dimensionality reduction, reduce to n_class-1 dimensions
    clf_lda = LDA()
    clf_lda.fit(x_train, y_train)
    x_train = clf_lda.transform(x_train)
    x_test = clf_lda.transform(x_test)

    print x_train.shape
    print x_test.shape
    '''

    print "Random Forest Model Learning..."

    #tree number in the random forest
    tree_num = 100
    #Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=tree_num, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features='auto', n_jobs=-1)
    clf.fit(x_train,y_train)

    print "Model Prediction..."
    #y_test = clf.predict(x_test)
    #convert y_test into 0-1 matrix
    #y_matrix = transform_output(y_test)

    #predict probability
    y_prob = clf.predict_proba(x_test)

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #the classe labels in encoder is consistent with the class labels in the classifier
        assert (encoder.classes_ == clf.classes_).all()
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))

    '''
    print clf.classes_
    print clf.feature_importances_
    index = np.argsort(clf.feature_importances_)
    print index
    print clf.feature_importances_[index]
    print y_test.shape
    print y_test[:10]
    '''

def logisticRegression(x_train, y_train, x_test, y_test):
    global get_test
    '''
    #use LDA to do dimensionality reduction, reduce to n_class-1 dimensions
    clf_lda = LDA()
    clf_lda.fit(x_train, y_train)
    x_train = clf_lda.transform(x_train)
    x_test = clf_lda.transform(x_test)

    print x_train.shape
    print x_test.shape
    '''
    '''
    #use PCA to do dimensionality reduction
    pca = PCA(n_components=8)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    #print x_train.shape
    #print x_test.shape
    #print x_train[:3, :]
    #print x_test[:3, :]
    '''


    for c_value in [0.8]:
        print "C value is: " + str(c_value)

        print "Logistic Regression Model Learning..."

        start_time = time.time()
        #logistic regression
        #solver: 'newton-cg', 'lbfgs', 'liblinear'
        #multi_class: 'ovr', 'multinomial'
        #in this problem, we have multiple class, 'multinomial' supports the real multi-class classifier, 'ovr' is One-VS-Rest
        #'multinomial' is only supported by 'newton-cg' and 'lbfgs' solver, and these two solvers only support 'l2' penalty
        #'liblinear' solver supports both 'l1' and 'l2' penalty, and 'l1' penalty can achieve sparsity and feature selection
        #possible combination:
        #1. penalty='l2', solver='liblinear', multi_class='ovr'
        #2. penalty='l1', solver='liblinear', multi_class='ovr'
        #3. penalty='l2', solver='lbfgs', multi_class='ovr'
        #4. penalty='l2', solver='lbfgs', multi_class='multinomial'
        #support from sklearn '0.16'
        #clf = linear_model.LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', multi_class='multinomial')
        #for sklearn '0.14' or '0.15', we can only use 'liblinear' and 'ovr' as following:
        clf = linear_model.LogisticRegression(penalty='l2', C=c_value)
        clf.fit(x_train,y_train)
        learning_time = time.time() - start_time
        print "training time is: {:.5f} seconds.".format(learning_time)

        print "Model Prediction..."
        #y_test = clf.predict(x_test)
        #convert y_test into 0-1 matrix
        #y_matrix = transform_output(y_test)

        start_time = time.time()
        #get probability prediction
        y_prob = clf.predict_proba(x_test)

        prediction_time = time.time() - start_time
        print "prediction time is: {:.5f} seconds.".format(prediction_time)

        if get_test == True:
            #the data is from real test set
            #output to file
            output_result(y_prob)
        else:
            #the test set is split from the train set, compute the loss function value
            encoder = LabelEncoder()
            #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
            y_true = encoder.fit_transform(y_test)
            #the classe labels in encoder is consistent with the class labels in the classifier
            assert (encoder.classes_ == clf.classes_).all()
            #compute the value for loss function
            score = logloss_mc(y_true, y_prob)
            print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))

        #print y_test.shape
        #print y_test[:10]


def naiveBayes(x_train, y_train, x_test, y_test):
    global get_test

    print "Multinomial Naive Bayes Model Learning..."

    #we use multinomial Naive Bayes here, because the features are counts of events, which are non-negative integers
    #there are no parameters to tune here

    start_time = time.time()
    #alpha control the Laplace smoothing extent, the larger alpha, the more smooth
    clf = MultinomialNB(alpha=1.0)
    clf.fit(x_train, y_train)

    learning_time = time.time() - start_time
    print "training time is: {:.5f} seconds.".format(learning_time)

    print "Model Prediction..."
    #y_predict = clf.predict(x_test)

    start_time = time.time()
    #get probability prediction
    y_prob = clf.predict_proba(x_test)

    prediction_time = time.time() - start_time
    print "prediction time is: {:.5f} seconds.".format(prediction_time)

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #the classe labels in encoder is consistent with the class labels in the classifier
        assert (encoder.classes_ == clf.classes_).all()
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))

    #print y_test[:10]
    #print y_predict[:10]

def lda_model(x_train, y_train, x_test, y_test):
    global get_test

    print "LDA model learning..."

    start_time = time.time()
    #LDA assumes common variance matrix among classes, while QDA doesn't
    clf = LDA()
    #clf = QDA()
    clf.fit(x_train, y_train)

    learning_time = time.time() - start_time
    print "training time is: {:.5f} seconds.".format(learning_time)

    '''
    #use LDA to do dimensionality reduction, reduce to n_class-1 dimensions
    x_t = clf.transform(x_train)
    print x_train.shape
    print x_t.shape
    print x_train[:3]
    print x_t[:3]
    '''

    print "Model Prediction..."
    #y_predict = clf.predict(x_test)

    start_time = time.time()
    #get probability prediction
    y_prob = clf.predict_proba(x_test)

    prediction_time = time.time() - start_time
    print "prediction time is: {:.5f} seconds.".format(prediction_time)

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #the classe labels in encoder is consistent with the class labels in the classifier
        assert (encoder.classes_ == clf.classes_).all()
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))

def stack(x_train, y_train, x_test, y_test):
    global get_test

    print "stack several classifier..."
    '''
    #LDA
    print "LDA..."
    clf1 = LDA()
    clf1.fit(x_train, y_train)
    y_prob1 = clf1.predict_proba(x_test)
    '''
    #LR
    print "Logistic Regression..."
    clf2 = linear_model.LogisticRegression(penalty='l2', C=1.0)
    clf2.fit(x_train,y_train)
    y_prob2 = clf2.predict_proba(x_test)

    #random forest
    print "Random Forest..."
    tree_num = 100
    #Random Forest Classifier
    clf3 = RandomForestClassifier(n_estimators=tree_num, n_jobs=-1)
    clf3.fit(x_train,y_train)
    y_prob3 = clf3.predict_proba(x_test)

    print "stacking..."
    #y_prob = (y_prob1 + y_prob2 + y_prob3)/3
    y_prob = (y_prob2 + y_prob3)/2

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #the classe labels in encoder is consistent with the class labels in the classifier
        #assert (encoder.classes_ == clf1.classes_).all()
        assert (encoder.classes_ == clf2.classes_).all()
        assert (encoder.classes_ == clf3.classes_).all()
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))

def bagging(x_train, y_train, x_test, y_test):
    global get_test
    #use bagging of base classifiers, like logistic regression
    print "Bagging classifier..."

    #bagging with LR models
    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    #number of models in the bagging
    n_model = 70
    #get bagging model
    clf = BaggingClassifier(base_estimator=model, n_estimators=n_model, max_samples=1.0, max_features=1.0, n_jobs=-1, random_state=1)
    #fit bagging model
    clf.fit(x_train, y_train)
    #prediction probability
    y_prob = clf.predict_proba(x_test)

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #the classe labels in encoder is consistent with the class labels in the classifier
        assert (encoder.classes_ == clf.classes_).all()
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))


def boosting(x_train, y_train, x_test, y_test):
    #use AdaBoost and GradientTreeBoosting
    print "AdaBoost..."

    #boosting with DT models
    model = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_samples_leaf=10, max_features=1.0, random_state=0)
    #number of models in the bagging
    n_model = 50
    #AdaBoost model
    clf = AdaBoostClassifier(base_estimator=model, n_estimators=n_model, learning_rate=1.0, algorithm='SAMME.R', random_state=1)
    #fit AdaBoost model
    clf.fit(x_train, y_train)
    #prediction probability
    y_prob = clf.predict_proba(x_test)

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #the classe labels in encoder is consistent with the class labels in the classifier
        assert (encoder.classes_ == clf.classes_).all()
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))


def neuralNetwork(x_train, y_train, x_test, y_test):
    global get_test

    #use neural network to classifier
    print "Neural Network Model Learning..."
    num_train_case = x_train.shape[0]
    #get random permutation index
    index = np.random.permutation(num_train_case)
    #random permute x_train and y_train
    x_train = x_train[index,:]
    y_train = y_train[index]
    #encode y
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train).astype(np.int32)
    #print encoder.classes_
    #scale x_train
    #scaler = StandardScaler()
    #x_train = scaler.fit_transform(x_train)
    #scale x_test
    #x_test = scaler.transform(x_test.astype(np.float32))

    num_classes = len(encoder.classes_)
    num_features = x_train.shape[1]

    #Train Neural Net

    layers0 = [ ('input', InputLayer),
                ('dense0', DenseLayer),
                ('dropout', DropoutLayer),
                ('dense1', DenseLayer),
                ('output', DenseLayer)]

    net0 = NeuralNet(   layers=layers0,
                        input_shape=(None, num_features),
                        dense0_num_units=2000,
                        dropout_p=0.5,
                        dense1_num_units=200,
                        output_num_units=num_classes,
                        output_nonlinearity=softmax,
                        update=nesterov_momentum,
                        update_learning_rate=0.01,
                        update_momentum=0.9,
                        eval_size=0.2,
                        verbose=1,
                        max_epochs=20)

    '''
    epochs = []

    def on_epoch_finished(nn, train_history):
        epochs[:] = train_history
        if len(epochs) > 1:
            raise StopIteration()

    net0 = NeuralNet(
        layers=[
            ('input', InputLayer),
            ('hidden1', DenseLayer),
            ('dropout1', DropoutLayer),
            ('hidden2', DenseLayer),
            #('dropout2', DropoutLayer),
            ('output', DenseLayer),
            ],
        input_shape=(None, num_features),
        output_num_units=num_classes,
        output_nonlinearity=softmax,
        hidden1_num_units=200,
        hidden2_num_units=200,
        dropout_p=0.5,
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=20,
        on_epoch_finished=on_epoch_finished,
        )

    #net0 = clone(nn_def)
    '''

    #model learning
    net0.fit(x_train, y_train)
    #probability prediction
    y_prob = net0.predict_proba(x_test)

    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value

        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.transform(y_test).astype(np.int32)
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))


def SVM(x_train, y_train, x_test, y_test):
    global get_test

    print "SVM Model Learning..."

    '''
    #use LinearSVC, which is based on liblinear
    clf = svm.LinearSVC(penalty='l2', loss='l1', C=1.0, multi_class='ovr')
    clf.fit(x_train, y_train)

    y_prob = clf.decision_function(x_test)
    y_prob = y_prob + 1000
    '''

    #The implementation is based on libsvm.
    #The fit time complexity is more than quadratic with the number of samples...
    # which makes it hard to scale to dataset with more than a couple of 10000 samples.
    #SVM original implementation is not probabilistic

    #SVM with default rbf kernel
    #clf = svm.SVC(C=1.0, kernel='rbf', gamma=0.0, probability=True)
    clf = svm.SVC(C=1.0, kernel='poly', gamma=0.0, probability=True, verbose=True)
    clf.fit(x_train, y_train)

    print "Model Prediction..."
    #y_test = clf.predict(x_test)
    #convert y_test into 0-1 matrix
    #y_matrix = transform_output(y_test)

    #get probability prediction
    y_prob = clf.predict_proba(x_test)


    if get_test == True:
        #the data is from real test set
        #output to file
        output_result(y_prob)
    else:
        #the test set is split from the train set, compute the loss function value
        encoder = LabelEncoder()
        #encode string label 'Class_1', 'Class_2',... to [0,1,...,8]
        y_true = encoder.fit_transform(y_test)
        #the classe labels in encoder is consistent with the class labels in the classifier
        assert (encoder.classes_ == clf.classes_).all()
        #compute the value for loss function
        score = logloss_mc(y_true, y_prob)
        print(" -- Multiclass logloss on validation set: {:.5f}.".format(score))

    #print y_test.shape
    #print y_test[:10]


def main():
    global get_test
    #s = np.array(['Class_4', 'Class_6', 'Class_6', 'Class_2', 'Class_1'])
    #print transform_output(s)

    print "Get Data..."

    #specify whether to read test data
    #if False, then don't get test data, use part of train data as test data
    #if True, get test data, and output prediction result to upload to Kaggle

    x, y, x_test = get_data()
    #print x.shape
    #(61878, 93)
    #print y.shape
    #(61878,)
    #print x_test.shape
    #(144368, 93)
    '''
    bin_num=10
    range_min=0
    range_max=10
    x_trans = x.reshape(-1, 1)
    #print x_trans.shape
    plot_hist(x[:, 60], bin_num, range_min, range_max)
    '''
    #print the ratio of class in the dataset
    #print_class_ratio(y)
    '''
    data_ratio = 1.0
    num_data_case = x.shape[0]
    #get random permutation index
    index = np.random.permutation(num_data_case)
    #random permute x_train and y_train
    x = x[index,:]
    y = y[index]
    #select a ratio of data case
    upper_bound = min(int(num_data_case*data_ratio), num_data_case+1)
    x = x[:upper_bound, :]
    y = y[:upper_bound]
    '''
    if get_test == False:
        #the test set is split from the train set
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=0)
        #print x_train.shape, y_train.shape
        #print x_test.shape, y_test.shape
        #print x_train[:3,:]
        #print y_train[:3]
        #print x_test[:3,:]
        #print y_test[:3]
    else:
        #the test set is the real test set
        x_train = x
        y_train = y
        x_test = x_test
        y_test = []


    '''
    #scaling the training and test feature data
    #--- not useful, testest for LR, randomforest, MultinomialNB cannot use negative values
    scaler = StandardScaler().fit(x_train*1.0)
    x_train = scaler.transform(x_train*1.0)
    x_test = scaler.transform(x_test*1.0)
    '''

    #1. uniform probability benchmark
    #uniform_prob_benchmark(x_test, y_test)

    #2. random forest
    #randomForest(x_train, y_train, x_test, y_test)

    #3. logistic regression
    #logisticRegression(x_train, y_train, x_test, y_test)

    #4. SVM  -- give up, not suitable
    #permutate x and y, and select 5000 to train SVM, 100 to test
    #index = np.random.permutation(len(y_train))
    #x_train = x_train[index,:]
    #y_train = y_train[index]
    #print np.amin(x_train)
    #print np.amin(x_test)
    #SVM(x_train[:5000,:], y_train[:5000], x_test[:100,:])
    #SVM(x_train, y_train, x_test, y_test)

    #5. Multinomial Naive Bayest
    #naiveBayes(x_train, y_train, x_test, y_test)

    #6. LDA
    #lda_model(x_train, y_train, x_test, y_test)

    #7. stacking
    #stack(x_train, y_train, x_test, y_test)

    #8. bagging -- support from sklearn 0.15
    bagging(x_train, y_train, x_test, y_test)

    #9. boosting
    #boosting(x_train, y_train, x_test, y_test)

    #10. Neural Network
    #neuralNetwork(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
