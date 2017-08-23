import sys
import pickle
import pprint
import matplotlib.pyplot 
sys.path.append("../tools/")

from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'pct_poi_inbound',
                 'pct_poi_outbound'] 
                 
financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 
                      'long_term_incentive', 'restricted_stock', 
                      'director_fees']
                      
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']
email_features.remove('email_address')

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

#pprint.pprint(data_dict['TOTAL'])
data_dict.pop('TOTAL', 0)

# Do any records have no financial data? If so, no action is required
# featureFormat takes care of that
for person in data_dict.keys():
    count = 0
    for value in data_dict[person].values():
        if value != 'NaN' and value != False:
            count += 1
    if count == 0:
        print person, 'has no data'

### Delete THE TRAVEL AGENCY IN THE PARK as an outlier
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

def compute_fraction( numerator, denominator ):
    if numerator == 'NaN' or denominator == 'NaN':
        fraction = 0
    else:
        fraction = float(numerator)/float(denominator)    
    return round(fraction, 2)

def add_fraction_to_dict(dict, numerator, denominator, new_variable_name):
    num = dict[numerator]
    den = dict[denominator]
    fraction = compute_fraction(num, den)
    dict[new_variable_name] = fraction
    return dict

for p in my_dataset:   
    # Calculate inbound POI email fraction
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
                                        'from_poi_to_this_person', 
                                        'to_messages',
                                        'fraction_from_poi')
        
    # Calculate outbound POI email fraction
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
                                        'from_this_person_to_poi',
                                        'from_messages',
                                        'fraction_to_poi')   
    
    # Calculate Salary as fraction of Total Payments
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
                                        'salary',
                                        'total_payments',
                                        'fraction_salary_total_payments') 

email_features = email_features + ['fraction_from_poi', 'fraction_to_poi']
financial_features = financial_features + ['fraction_salary_total_payments']
                  
features_list = ['poi'] + financial_features + email_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
# Split the data into training and testing sets
# features_train, features_test, labels_train, labels_test =  train_test_split(
#   features,
#   labels,
#   test_size=0.2,
#   random_state=42
#   )

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Make an StratifiedShuffleSplit iterator for cross-validation in GridSearchCV
sss = StratifiedShuffleSplit(labels, 100, random_state=42)

### NaiveBayes
skb = SelectKBest(f_classif)
pca = PCA()
clf_nb = GaussianNB()

pipe = Pipeline(steps=[("skb", skb), 
                      ("pca", pca), 
                      ("clf", clf_nb)])

params = {"pca__n_components":[2, 3, 4, 5, 6, 7, 8],
          "skb__k":[8, 9, 10, 11, 12]}

# caculate run time
def get_time(time1, time2, name):
    delta = time2 - time1
    mins = delta.seconds/60
    secs  = delta.seconds%60
    print '###########################################################'
    print 'Model',name, 'started from', time1, 'ended at', time2
    print 'It took', mins, 'minutes',secs, 'seconds','to run the model'
    print '###########################################################'

# fit the model and get the run time
def fit_model(pipe, params, name):
    time1 = datetime.now() 
    clf = GridSearchCV(pipe, params, scoring='f1', cv=sss)
    clf.fit(features, labels)
    time2 = datetime.now()
    get_time(time1, time2, name)
    return clf

# Fit the model using premade estimator clf
clf = fit_model(pipe, params, 'NB')

# get selected feature list
features_selected_bool = clf.best_estimator_.named_steps['skb'].get_support()
features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
features_selected_list
#clf.best_params_

# get selected features, scores, p-values:
def get_selected_features_scores_pvalues(clf):
    ## print Selected Features, Scores, and P-Values
    X_new = clf.best_estimator_.named_steps['skb']
    # Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
    feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
    # Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
    feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
    # Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
    # create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
    features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

    # Sort the tuple by score, in reverse order
    features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

    # Print
    print ' '
    print 'Selected Features, Scores, P-Values:'
    print features_selected_tuple

get_selected_features_scores_pvalues(clf)

# test the best_estimator
test_classifier(clf.best_estimator_, my_dataset, features_list)


### KNN
# use MinMaxScaler or StandardScaler for feature scaling
#scaler = MinMaxScaler()
scaler = StandardScaler()

clf_knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('scaler', scaler),
                       ("skb", skb), 
                       ("pca", pca), 
                       ('knn', clf_knn)])
params = {"skb__k":[7, 8, 9, 10, 11, 12],
          "pca__n_components":[2, 3, 4, 5],
          'knn__n_neighbors': [3, 5, 7],
          #'knn__weights': ['uniform', 'distance']
          }

clf = fit_model(pipe, params, 'knn')

#features_selected_bool = clf.best_estimator_.named_steps['skb'].get_support()
#features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
#features_selected_list
#clf.best_params_

get_selected_features_scores_pvalues(clf)

test_classifier(clf.best_estimator_, my_dataset, features_list)

### DecisionTree
clf_tree = DecisionTreeClassifier(random_state = 42)
pipe = Pipeline(steps=[("skb", skb), 
                       ("pca", pca), 
                       ('tree', clf_tree)])
params = {"pca__n_components":[2, 3, 4, 5, 6],
          "skb__k":[6, 7, 8, 9, 10, 11, 12],
          #'tree__criterion': ['gini', 'entropy'],
          #'tree__splitter': ['best', 'random'],
          'tree__min_samples_split': [3, 4, 5, 6]}

clf = fit_model(pipe, params, 'tree')

#features_selected_bool = clf.best_estimator_.named_steps['skb'].get_support()
#features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
#features_selected_list
#clf.best_params_

get_selected_features_scores_pvalues(clf)

test_classifier(clf.best_estimator_, my_dataset, features_list)

# save this as the best clf for future testing
best_clf = clf.best_estimator_

### RandomForest
clf_rf = RandomForestClassifier()
pipe = Pipeline(steps=[("skb", skb), 
                       #("pca", pca), 
                       ('rf', clf_rf)])
params = {"skb__k":[6, 7, 8, 9, 10, 11],
          #'rf__criterion': ['gini', 'entropy'],
          'rf__n_estimators': [2, 5, 7, 10],
          #'rf__max_features': ['auto', 'sqrt', 'log2', None]
          }

clf = fit_model(pipe, params, 'rf')

#features_selected_bool = clf.best_estimator_.named_steps['skb'].get_support()
#features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
#features_selected_list
#clf.best_params_

get_selected_features_scores_pvalues(clf)

test_classifier(clf.best_estimator_, my_dataset, features_list)

### SVM
svm = SVC(random_state = 42)
pipe = Pipeline(steps=[('scaler', scaler), ("SKB", skb), ("PCA", pca), ('SVM', svm)])
params = {"PCA__n_components":[2, 3, 4, 5],
          "SKB__k":[6, 7, 8, 9],
          'SVM__C': [0.1, 1, 10, 100, 1000, 5000],
          #'SVM__kernel': ['rbf', 'poly'],
          #'SVM__gamma': [0.01, 0.1, 1, 2, 'auto'],
          }

clf = fit_model(pipe, params, 'svm')

#features_selected_bool = clf.best_estimator_.named_steps['skb'].get_support()
#features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
#features_selected_list
#clf.best_params_

test_classifier(clf.best_estimator_, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(best_clf, my_dataset, features_list)