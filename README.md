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
![PCA-2D](https://user-images.githubusercontent.com/75994180/227760603-8f23d427-455c-4edf-a7b2-5a1ec844c84e.png)
