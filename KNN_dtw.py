# %%
from sklearn.model_selection import cross_val_predict
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw_ndim
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import json
import pandas as pd
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

with open('ae_train.json') as json_file:
    data_train = json.load(json_file)

with open('ae_test.json') as json_file:
    data_test = json.load(json_file)

# %% example of DTW
dat = pd.DataFrame.from_dict(data_train)

time1 = np.linspace(0, 20, 20)
time2 = np.linspace(0, 30, 30)
timeseries1 = np.sin(time1)
timeseries2 = 1.4*np.sin(time2)
for i, value in enumerate(timeseries2):
    if i % 2 == 0:
        timeseries2[i] = value + np.random.normal()/5

distance = dtw(timeseries1, timeseries2)
dtwpath = dtw_path(timeseries1, timeseries2)

fig = plt.figure(figsize=(12, 4))
plt.plot(time1, timeseries1, label='A')
plt.plot(time2, timeseries2, label='B')

plt.title('DTW distance between A and B is %.2f' % distance)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.legend()
# %% first attempt

x_train = []
y_train = []
x_test = []
y_test = []

# sort the data into a numpy arrays
for speaker in data_train:
    for recording in data_train[speaker]:
        # XXX JUST FOR TESTING
        x_train.append(np.array(data_train[speaker][recording]))
        y_train.append(speaker)

for speaker in data_test:
    for recording in data_test[speaker]:
        # XXX JUST FOR TESTING
        x_test.append(np.array(data_test[speaker][recording]))
        y_test.append(speaker)

# sort this into a 9x12x370 array (if that's what you want)
#x_train = np.dstack(x_train)
#y_train = np.dstack(y_train)

# extract a single channel because I don't know how to do it on a n x m array
chan1_train = []
chan1_test = []
for i in x_train:
    chan1_train.append(i[-1])

for i in x_test:
    chan1_test.append(i[-1])

# convert to proper arrays
chan1_test = np.array(chan1_test)
chan1_train = np.array(chan1_train)
y_test = np.array(y_test)
# np.dstack(chan1_train)
# try out all of these neighbors
parameters = {'n_neighbors': [7, 8, 9, 16]}  # 8

clf = GridSearchCV(KNeighborsClassifier(
    metric=dtw, weights='distance'), parameters, verbose=1, cv=5)
clf.fit(chan1_train, y_train)
y_pred = clf.predict(chan1_test)
print(classification_report(y_test, y_pred))

# 0.90
# %% second attempt using ndim

x_train = []
y_train = []
x_test = []
y_test = []

# sort the data into a numpy arrays
for speaker in data_train:
    for recording in data_train[speaker]:
        # XXX JUST FOR TESTING
        x_train.append(np.array(data_train[speaker][recording]))
        y_train.append(speaker)

for speaker in data_test:
    for recording in data_test[speaker]:
        # XXX JUST FOR TESTING
        x_test.append(np.array(data_test[speaker][recording]))
        y_test.append(speaker)

# convert to proper arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# try out all of these neighbors
parameters = {'n_neighbors': [7, 8, 9, 16]}  # 8

clf = GridSearchCV(KNeighborsClassifier(
    metric=dtw_ndim.distance, weights='distance'), parameters, verbose=1, cv=5)
clf.fit(chan1_train, y_train)
y_pred = clf.predict(chan1_test)
print(classification_report(y_test, y_pred))

# 0.90
# %% third attempt: pca

from sklearn.decomposition import PCA
pca = PCA(n_components=1)

new_x_train = []
for x in x_train:
    pc = pca.fit_transform(x)
    new_x_train.append(pc)

new_x_test = []
for x in x_test:
    pc = pca.fit_transform(x)
    new_x_test.append(pc)

new_x_test = np.vstack(new_x_test)
new_x_train = np.squeeze(new_x_train)

# FIXME: doesn't work yet but I had to go, sorry peeps

parameters = {'n_neighbors': [7, 8, 9, 16]}  # 8
clf = GridSearchCV(KNeighborsClassifier(
    metric=dtw, weights='distance'), parameters, verbose=1, cv=5)
clf.fit(new_x_train, y_train)
y_pred = clf.predict(chan1_test)
print(classification_report(y_test, y_pred))

# %%
