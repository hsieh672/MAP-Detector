import csv
import numpy as np
import pandas as pd
import random
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

train_num = 484
test_num = 20
type_num = 3

# read the csv file 
fn = 'C:/Users/88697/Desktop/NTHU/ML/HW1/Wine.csv'
with open(fn) as csvFile : 
    csvReader = pd.read_csv(csvFile,sep = ',',header = None) 
data = np.array(csvReader)
print(data)

# count the number of different types
count_type = [0,0,0]
for i in range(1,csvReader.shape[0]):
    if int(csvReader[0][i]) == 0 : 
        count_type[0] += 1
    elif int(csvReader[0][i]) == 1 :
        count_type[1] += 1
    elif int(csvReader[0][i]) == 2 :
        count_type[2] += 1

test_type = [[0]*csvReader.shape[1] for i in range(type_num)]
test_type[0] = [random.randint(1,count_type[0])for _ in range(test_num)]
test_type[1] = [random.randint(count_type[0] + 1,count_type[0] + count_type[1])for _ in range(test_num)]
test_type[2] = [random.randint(count_type[0] + count_type[1] + 1,
                               count_type[0] + count_type[1] + count_type[2])for _ in range(test_num)]

# save the testing and training data into csv files
f_test = 'C:/Users/88697/Desktop/NTHU/ML/HW1/testing.csv'
f_train = 'C:/Users/88697/Desktop/NTHU/ML/HW1/training.csv'

sum = 0
data_test = [[0]*csvReader.shape[1] for i in range(type_num*test_num)]

with open(f_test,'w+',newline='') as f :
    writer = csv.writer(f)
    for j in range(type_num) :
        for i in range(test_num) :
            writer.writerow(data[test_type[j][i]]+' ')
            data_test[sum] = data[test_type[j][i]]
            sum += 1

sum = 0
data_train = [[0]*csvReader.shape[1] for i in range(train_num-1)]
with open(f_train,'w+',newline='') as f :
    writer = csv.writer(f)
    for i in range(1,train_num) :
        writer.writerow(data[i]+' ')
        data_train[sum] = data[i]
        sum += 1

# delete the label of the input data
data_train = np.array(data_train)
data_train = np.asarray(data_train, dtype=float)
data_test = np.array(data_test)
data_test = np.asarray(data_test, dtype=float)
data_feature = np.delete(data_train, 0, 1)
data_feature_test = np.delete(data_test, 0, 1)

# split the training data based on their features
data_train_0 = ([[0]*(csvReader.shape[1]-1) for i in range(count_type[0])])
data_train_1 = [[0]*(csvReader.shape[1]-1) for i in range(count_type[1])]
data_train_2 = [[0]*(csvReader.shape[1]-1) for i in range(count_type[2])]

for i in range(0,count_type[0]):
    data_train_0[i] = data_feature[i]
for i in range(0,count_type[1]):
    data_train_1[i] = data_feature[count_type[0] + i]
for i in range(0,count_type[2]):
    data_train_2[i] = data_feature[count_type[0] + count_type[1] + i]

# calculate the mean and std of each features
feature_mean_0 = [[0]*(csvReader.shape[1]-1)]
feature_mean_1 = [[0]*(csvReader.shape[1]-1)]
feature_mean_2 = [[0]*(csvReader.shape[1]-1)]
feature_std_0 = [[0]*(csvReader.shape[1]-1)]
feature_std_1 = [[0]*(csvReader.shape[1]-1)]
feature_std_2 = [[0]*(csvReader.shape[1]-1)]

feature_mean_0 = np.mean(data_train_0,axis = 0)
feature_mean_1 = np.mean(data_train_1,axis = 0)
feature_mean_2 = np.mean(data_train_2,axis = 0)
feature_std_0 = np.std(data_train_0,axis = 0)
feature_std_1 = np.std(data_train_1,axis = 0)
feature_std_2 = np.std(data_train_2,axis = 0)

# calculate prior probability
prior = [0.,0.,0.]
for i in range(type_num):
    prior[i] = count_type[i] / train_num
print(prior)

# calculate multiplication of likelihood and prior probability
delta = 1e-3
post = [0.,0.,0.]
post_0 = ([1.]*(test_num*type_num))
post_1 = ([1.]*(test_num*type_num))
post_2 = ([1.]*(test_num*type_num))
label = ([0]*(test_num*type_num))


for i in range(test_num*type_num):
    post_0[i] = post_0[i] * prior[0]
    post_1[i] = post_1[i] * prior[1]
    post_2[i] = post_2[i] * prior[2]
    for j in range(csvReader.shape[1]-1):
        distribution_0 = st.norm(feature_mean_0[j],feature_std_0[j])
        distribution_1 = st.norm(feature_mean_1[j],feature_std_1[j])
        distribution_2 = st.norm(feature_mean_2[j],feature_std_2[j])
        likelihood_0 = integrate.quad(distribution_0.pdf,data_feature_test[i][j], data_feature_test[i][j]+delta)
        likelihood_1 = integrate.quad(distribution_1.pdf,data_feature_test[i][j], data_feature_test[i][j]+delta)
        likelihood_2 = integrate.quad(distribution_2.pdf,data_feature_test[i][j], data_feature_test[i][j]+delta)
        post_0[i] = likelihood_0[0] * post_0[i]
        post_1[i] = likelihood_1[0] * post_1[i]
        post_2[i] = likelihood_2[0] * post_2[i]
    post = [post_0[i],post_1[i],post_2[i]]
    label[i] = np.argmax(post)

# calculate the accuracy rate of MAP detection
correct = 0

for i in range(test_num*type_num):
    if label[i] == data_test[i][0]:
         correct += 1
    accuracy = correct / (test_num*type_num)
print('accuracy use all features in ML: ',accuracy)



# plot the visualized result of testing data
PCA2 = PCA(n_components=2)
PCA3 = PCA(n_components=3)
x2 = PCA2.fit(data_feature).transform(data_feature)
x3 = PCA3.fit(data_feature).transform(data_feature)
y = data_train[:,0]

markers = ['v', 's', 'o']
wines = ['type 0', 'type 1', 'type 2']
labels = [0., 1., 2.]

# plot 2D
for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt.scatter(x2[y==i, 0], x2[y==i, 1], c=c, label=target_name, marker=m)
plt.xlabel('PCA-feature-1')
plt.ylabel('PCA-feature-2')
plt.legend(wines ,loc='upper right')
plt.savefig('C:/Users/88697/Desktop/NTHU/ML/HW1/PCA-2D.png')

plt.clf()

#plot 3D
plt3 = plt.axes(projection='3d')
for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt3.scatter(x3[y==i, 0], x3[y==i, 1],x3[y==i, 2], c=c, label=target_name, marker=m)
plt3.set_xlabel('PCA-feature-1')
plt3.set_ylabel('PCA-feature-2')
plt3.set_zlabel('PCA-feature-3')
plt.legend(wines ,loc='upper right')
plt.savefig('C:/Users/88697/Desktop/NTHU/ML/HW1/PCA-3D.png')

# correlation between PCA2 and test dada and PCA3 and test data

PCA2_col1 = np.array(PCA2.components_[0,:])
PCA2_col2 = np.array(PCA2.components_[1,:])
PCA3_col1 = np.array(PCA3.components_[0,:])
PCA3_col2 = np.array(PCA3.components_[1,:])
PCA3_col3 = np.array(PCA3.components_[2,:])

PCA2_components = pd.DataFrame({'PCA-feature-1' : PCA2_col1,
                                'PCA-feature-2' : PCA2_col2},
                                index = ['1','2','3','4','5','6','7','8','9','10','11','12','13'])
PCA3_components = pd.DataFrame({'PCA-feature-1' : PCA3_col1,
                                'PCA-feature-2' : PCA3_col2,
                                'PCA-feature-3' : PCA3_col3},
                                index = ['1','2','3','4','5','6','7','8','9','10','11','12','13'])

# save PCA data into csv file
f_PCA2 = 'C:/Users/88697/Desktop/NTHU/ML/HW1/PCA2_components.csv'
f_PCA3 = 'C:/Users/88697/Desktop/NTHU/ML/HW1/PCA3_components.csv'

PCA2_components.to_csv(f_PCA2,index=False)
PCA3_components.to_csv(f_PCA3,index=False)

# the 5th and 13th data in PCA2 components dominate the features 
# the 4th, 5th and 13th data in PCA3 components dominate the features 
# use the labels above to predict the type of wine
delta = 1e-3
post = [0.,0.,0.]
post_0 = ([1.]*(test_num*type_num))
post_1 = ([1.]*(test_num*type_num))
post_2 = ([1.]*(test_num*type_num))
label = ([0]*(test_num*type_num))

for i in range(test_num*type_num):
    post_0[i] = post_0[i] * prior[0]
    post_1[i] = post_0[i] * prior[1]
    post_2[i] = post_0[i] * prior[2]
    for j in range(csvReader.shape[1]-1):
        if j == 3 or j == 4 or j == 12:
            distribution_0 = st.norm(feature_mean_0[j],feature_std_0[j])
            distribution_1 = st.norm(feature_mean_1[j],feature_std_1[j])
            distribution_2 = st.norm(feature_mean_2[j],feature_std_2[j])
            likelihood_0 = integrate.quad(distribution_0.pdf,data_feature_test[i][j], data_feature_test[i][j]+delta)
            likelihood_1 = integrate.quad(distribution_1.pdf,data_feature_test[i][j], data_feature_test[i][j]+delta)
            likelihood_2 = integrate.quad(distribution_2.pdf,data_feature_test[i][j], data_feature_test[i][j]+delta)
            post_0[i] = likelihood_0[0] * post_0[i]
            post_1[i] = likelihood_1[0] * post_1[i]
            post_2[i] = likelihood_2[0] * post_2[i]
    post = [post_0[i],post_1[i],post_2[i]]
    label[i] = np.argmax(post)

# calculate the accuracy rate of MAP detection
correct = 0

for i in range(test_num*type_num):
    if label[i] == data_test[i][0]:
         correct += 1
    accuracy = correct / (test_num*type_num)
print('accuracy only use three features : ',accuracy)