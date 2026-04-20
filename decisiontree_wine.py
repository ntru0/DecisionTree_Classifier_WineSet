import numpy as np
import pandas as pd
 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# different data set name. uncomment and use one or the other.
FILENAMEPATH = 'wines.csv'
#FILENAMEPATH = 'wine.data'


#0 is running the actual test and writing the results to file.
# 1 is printing printing trees.
TESTMODE = 1
#parameter tweaks -----------------------------------------------
#important to note that sample size is 1-this variable
TESTSIZE_PERCENTAG=0.20

RANDSTATE=17 #not important. just a seed.
MIN_SAMPLE_SPLIT=2 #3
MAXDEPTH= 4 #5
#NSPLITS=5

#helper function for me to print my results so I can craft the report better
def fileprinter(acc1 ):
    with open('output_p3.txt', 'a') as file2:
        s2 = f"{MIN_SAMPLE_SPLIT} & {MAXDEPTH} & {acc1:.2f} " + r" \\" + r" \hline" +  "\n"
        file2.write(s2)
    file2.close()

#matlab printer tree for report 
#Followed a youtube video to implement this decision tree by scratch
def plot_decision_tree(node, x=0.5, y=1, layer=1, width=0.1, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 10))

    # Plot the current node
    if node.value is not None:  # If it's a leaf node
        ax.text(x, y, f"Leaf: {node.value}", horizontalalignment="center", verticalalignment="center", fontsize=10)
        ax.scatter(x, y, s=100, c='r', marker='o')
    else:  # If it's not a leaf 
        ax.text(x, y, f"X{node.feature_index} <= {node.threshold}", horizontalalignment="center", verticalalignment="center", fontsize=10)
        ax.scatter(x, y, s=100, c='g', marker='o')

    # Recur
    if node.left:
        ax.plot([x, x - width], [y - 0.1, y - 0.3], 'k-',  lw=2)  # Left connection
        plot_decision_tree(node.left, x - width, y - 0.3 , layer + 1, width / 2, ax=ax)

    if node.right:
        ax.plot([x, x + width], [y - 0.1, y - 0.3], 'k-', lw=2)  # Right connection
        plot_decision_tree(node.right, x + width, y - 0.3, layer + 1, width / 2, ax=ax)

    return ax

#our col names
col_names = ['Type', 'Alcohol', 'MalicAcid',
             'Ash', 'Alcalinity', 'Magnesium','Phenols','Flavanoids','Nonflavanoid',
             'Proanthocyanins','Colorlntensity','Hue','DilutedWines','Proline']

#can det if the first line of the file is a header or not if the line contains strings
def is_header(row):
    for val in row:
        if isinstance(val, str):
            return True  # is string, true!!!
    return False


#heper function that loads data
#will handle if it has a header or not just fine.
#wines.csv and wine.data
def load_data(file_path, col_names):
    # takes the first row and see if it contains string (is header)
    first_row = pd.read_csv(file_path, nrows=1, header=None).iloc[0]
    
    if is_header(first_row):
        # If the first row has strings, its a header
        data = pd.read_csv(file_path, header=0, sep=',', skip_blank_lines=True, na_values='?')
    else:
        # add col name if no header
        data = pd.read_csv(file_path, header=None, names=col_names, sep=',', skip_blank_lines=True, na_values='?')
    
    return data

# Load the data
data = load_data(FILENAMEPATH, col_names)

#data = pd.read_csv('wines.csv',skiprows=1, header=None ,names=col_names, skip_blank_lines=True, na_values='?', sep=',')
# NHINOTE
if TESTMODE==1:
    print(data.head(30))
    # data = pd.read_csv('wine.data',skiprows=1, header=None ,names=col_names, skip_blank_lines=True, na_values='?', sep=',')

#node class
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        #constructor for the node
        #tree node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        #leave node
        self.value = value

#tree class, straight from n. nerd's video!
#I heavily referenced the video, i claim none of the vid as my own!!
#thank you normalized nerd!
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        # constructor
        
        # init da root yo
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        # function to compute information gain 
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        # function to compute entropy 
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        # function to compute gini index 
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        # function to compute leaf node 
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        # function to print the tree 
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        # function to train the tree 
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        # function to predict new dataset 
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        # function to predict a single data point 
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        

# Train-Test Split
X = data.iloc[:,1:].values
Y = data.iloc[:,0].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TESTSIZE_PERCENTAG, random_state=RANDSTATE)


#make the tree
classifier = DecisionTreeClassifier(min_samples_split=MIN_SAMPLE_SPLIT, max_depth=MAXDEPTH)
classifier.fit(X_train, Y_train)

# NHINOTE
if TESTMODE==1:
    plot_decision_tree(classifier.root)
    plt.show()
    #classifier.print_tree()

#prediction
Y_pred = classifier.predict(X_test)

#get accuracy and print
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of Training:", accuracy)


#if TESTMODE == 2:
#    param_grid = {
#        'min_samples_split': [2, 3, 4],  # Example values, adjust based on your use case
#        'max_depth': [3, 4, 5]  # Example values, None means no limit on depth
#    }

#    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=NSPLITS, n_jobs=-1, verbose=1)
#    grid_search.fit(X_train, Y_train)
#    print("Best parameters : ", grid_search.best_params_)

#if TESTMODE==2:
if TESTMODE==0:
    fileprinter(accuracy)

