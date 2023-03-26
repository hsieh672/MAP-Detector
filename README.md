# MAP-detector
Implement the Maximum A Posteriori  probability (MAP) of the classifier for 60 instances of 3 types wines with 13 different features.  
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
[data](https://archive.ics.uci.edu/ml/datasets/Wine)
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

![PCA-2D](https://user-images.githubusercontent.com/75994180/227760603-8f23d427-455c-4edf-a7b2-5a1ec844c84e.png)
