from sklearn import svm
import linecache
import random

ratio = 0.8

samples = linecache.getlines('svm5b_sep.txt')
random.shuffle(samples)
train = samples[0:int(len(samples)*ratio)]
test = samples[int(len(samples)*ratio):len(samples)]
X = []
y = []
X_test = []
y_test = []

for sample in train:
    sample_array_o = sample.split('\t')
    sample_array = sample_array_o[0:len(sample_array_o)-1]
    sample_result = sample_array_o[-1]
    for element in range(0, len(sample_array)):
        sample_array[element] = float(sample_array[element])
    sample_result = int(sample_result)
    X.append(sample_array)
    y.append(sample_result)

for sample in test:
    sample_array_o = sample.split('\t')
    sample_array = sample_array_o[0:len(sample_array_o)-1]
    sample_result = sample_array_o[-1]
    for element in range(0, len(sample_array)):
        sample_array[element] = float(sample_array[element])
    sample_result = int(sample_result)
    X_test.append(sample_array)
    y_test.append(sample_result)

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

truth = 0

predicts = clf.predict(X_test)

for i in range(0, len(predicts)):
    if predicts[i] == y_test[i]:
        truth += 1

print truth/(len(predicts)+0.0)