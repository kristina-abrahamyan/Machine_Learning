# Machine Learning Questions

1. Are we going to be touching upon K-Nearest Neighbors algorithm?
2. I am not sure what Q8 is even asking: it is about the difference between L1 and L2 regularisation
3. Q11: Fourier transform? -- I am guessing it's somehow converting one signal into another? but I don't understand the specifics...
4. Q14: confused about the meaning of generative model
5. What is k-folds cross-validation (mentioned in Q15)
6. Q19: are they saying that we may end up in the situation where our "test" sample only includes values for the dataset that has 90% of the data in one class?
7. Will we talk about ensemble techniques? seems too abstract from the explanation in the article
8. Q22: regularization technique LASSO? what is this?
9. Q25: kernal trick?
10. What can Spark compare to based on what we learned?
11.  Q28: pseudo-code for a parallel implementation?
12. Hash tables: we used them in SQL? to link the tables using keys?
13. Data pipelines, what are they conceptually? what are we linking together?
14. Q37: Business model usage to find valuable data in a business. Where do you look for the business model?
15. Q46: Google training its self-driving cars using recaptcha?






# Some theory
### Creating artificial datasets using scikit-learn
from sklearn.datasets import make_classification

x,y = make_classification(n_samples = 200, n_classes = 2, n_features = 10, n_redundant = 0, random_state = 1)

Why would we need to do this?



### Creating the same scale for comparison of different features
- normalize()
- StandardScaler()



### Feature Selection
Discarding features that have low variance
from sklearn.feature_selection import VarianceThreshold
def remove_low_var(input_data, threshold = 0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data(input_data.columns[selection.get_support(indices = True)])
x = remove_low_variance(x, threshold = 0.01)
x



### Feature Engineering
Categorical features need to be specially encoded. This can be done using:
- nominal features (eg Los Angeles, Bangkok)
- Ordinal features (eg low, medium, high)

In Python this is done using:
- Pandas (get_dummies() function and map() method)
- Scikit-learn (OneHotEncoder(), OrdinalEncoder(), LabelBinarizer(), LabelEncoder() )



### Imputing missing data
- SimpleImputer() function and,
- IterativeImputer() function 
from the sklearn.impute sub-module



### Splitting the data into a training and test sets
from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
x_train.shape, x_test.shape



### Building Pipelines
Use Pipeline() function
