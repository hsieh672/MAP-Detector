# MAP-detector
Implement the Maximum A Posteriori  probability (MAP) of the classifier for 60 instances of 3 types wines with 13 different features.   
[Wine Data](https://archive.ics.uci.edu/ml/datasets/Wine) 
#### Information of each feature: 
1. Alcohol  
2. Malic acid  
3. Ash  
4. Alcalinity of ash  
5. Magnesium  
6. Total phenols  
7. Flavanoids  
8. Non Flavonoid phenols  
9. Proanthocyanins  
10. Color intensity  
11. Hue  
12. OD280/OD315 of diluted wines   
13. Proline  
 
##  Split the testing and training data
```sh
train_num = 484
test_num = 20
type_num = 3

# read the csv file 
fn = 'C:/Users/88697/Desktop/NTHU/ML/HW1/Wine.csv'
with open(fn) as csvFile : 
    csvReader = pd.read_csv(csvFile,sep = ',',header = None) 
data = np.array(csvReader)
print(data)
```
The variable “count_type” counts the total number of each type. The number of different types are [type 0 , type 1 , type2] = [175 , 205 , 103]. The varianle “test_type[i]” records 20 randomly generated numbers in each type.  

```sh
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
```
Because I don’t need to use the label to calculate the posterior probabilities, so I use the variables “data_feature” and “data_feature_test” to save the training and testing data which delete the first column of “data_train” and “data_test” sepsrstely. I thought this section may be redundant because I actually only need to skip the first column in the calculation to achieve.  

```sh
# delete the label of the input data
data_train = np.array(data_train)
data_train = np.asarray(data_train, dtype=float)
data_test = np.array(data_test)
data_test = np.asarray(data_test, dtype=float)
data_feature = np.delete(data_train, 0, 1)
data_feature_test = np.delete(data_test, 0, 1)
```
## Using MAP detector to predict the testing data
![MAP](https://user-images.githubusercontent.com/75994180/227760788-8126c56d-3a0a-4868-8261-b80e4ac6bfb3.png)  
The posterior probability would calculate by the above equation. We know that 𝑃(𝑐|𝑋) ∝ 𝑃(𝑥|𝑐) ∗ 𝑃(𝑐), so I only need to calculate the likelihood and the class prior probability.  

Because when calculating posterior probabilities, I need to calculate the mean and standard deviation of each feature. I use “data_train_0”, “data_train_1” and “data_train_2” to save the training data of each type.  
```sh
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
```
I use “feature_mean_i” and “feature_std_i” to save the mean and standard deviation of the ith type.  
```sh
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
```
The prior probabilities of each type are [type0 , type1 , type2] = [0.362, 0.424, 0.214]  
```sh
# calculate prior probability
prior = [0.,0.,0.]
for i in range(type_num):
    prior[i] = count_type[i] / train_num
print(prior)
```

In this section, I need to categorize each category. There have 60 testing data and I need to calculate the posterior of every testing data and then find out the biggest number of the posterior.  
At first, I multiplied the posterior and the prior of each type. The equation is 𝑃(𝑐|𝑋) = 1 ∗ 𝑃(𝑐)  
I used the variable “feauture_mean_i” and “feauture_std_i” to obtain the pdf of the Gaussian distribution. The likelihood of each testing data would be integrated by the following equation and then obtain the posterior, the integral interval I set is 𝑑𝑒𝑙𝑡𝑎 = 10−3:  
![likelihood](https://user-images.githubusercontent.com/75994180/227760881-6c413138-a918-42f1-97ca-761f9b2f0988.png)  
Finally, I would find out the biggest posterior of each type then sort the testing data to that type which has the biggest posterior  
```sh
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

```
I compared the original data type and the data type I used the MAP detection to obtain to calculate the total number of correction.  
```sh
# calculate the accuracy rate of MAP detection
correct = 0

for i in range(test_num*type_num):
    if label[i] == data_test[i][0]:
         correct += 1
    accuracy = correct / (test_num*type_num)
print('accuracy use all features in ML: ',accuracy)
```
### Accuracy = 0.9666666667  
## Plot the visualized result of testing data
PCA is a way for analyzing large datasets containing a high number of dimensions or features per observation. In this HW, the training data have 423*13 dimensions. When the dimensions of the datasets are large, we need to reduce the dimensions to avoid overfitting.  
We need to use PCA to reduce the dimensions, find out the principal features of the data and use these data to classify the three types. 

```sh
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
```
![PCA-2D](https://user-images.githubusercontent.com/75994180/227760603-8f23d427-455c-4edf-a7b2-5a1ec844c84e.png)  
![PCA-3D](https://user-images.githubusercontent.com/75994180/227760980-6f4a1e14-3726-4258-84c0-71bf2273020a.png)  
[PCA2_components.csv](https://github.com/hsieh672/MAP-detector/files/11070988/PCA2_components.csv)  
[PCA3_components.csv](https://github.com/hsieh672/MAP-detector/files/11070989/PCA3_components.csv)  

Use the 4th, 5th and 13th features to MAP detector then predict the type of wine and calculate the accuracy rate.  
### Accuracy = 0.766666666667
