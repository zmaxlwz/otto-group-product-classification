import numpy as np
import matplotlib.pyplot as plt

ratio = [20, 40, 60, 80, 100]
train_LR = [2.362658, 6.145502, 9.91361, 13.533592, 16.529352]
test_LR = [0.002792, 0.004962, 0.0073, 0.009666, 0.01232]
train_NB = [0.022816, 0.042062, 0.05835, 0.079276, 0.104474]
test_NB = [0.003948, 0.006844, 0.010378, 0.014692, 0.019168]
train_LDA = [0.320714, 0.690454, 1.104214, 1.391908, 1.717372]
test_LDA = [0.002858, 0.005, 0.008016, 0.010682, 0.013832]

'''
plt.plot(ratio, train_LR, 'bo-', label="LR")
plt.plot(ratio, train_NB, 'go-', label="NB")
plt.plot(ratio, train_LDA, 'ro-', label="LDA")
plt.xlabel('Data Size(%)')
plt.ylabel('Train Time(s)')
plt.title('Training Time for classifiers')
plt.legend(['LR', 'NB', 'LDA'], loc='upper left')
plt.xlim([10,110])
plt.ylim([0, 18])
plt.show()
'''

plt.plot(ratio, test_LR, 'bo-', label="LR")
plt.plot(ratio, test_NB, 'go-', label="NB")
plt.plot(ratio, test_LDA, 'ro-', label="LDA")
plt.xlabel('Data Size(%)')
plt.ylabel('Prediction Time(s)')
plt.title('Prediction Time for classifiers')
plt.legend(['LR', 'NB', 'LDA'], loc='upper left')
plt.xlim([10,110])
plt.ylim([0, 0.05])
plt.show()
