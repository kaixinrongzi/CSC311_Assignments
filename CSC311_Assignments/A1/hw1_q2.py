import sklearn
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from numpy import random
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def my_tokenize(line):    #line is a list of words
    return line.split()

# vectorizer.get_feature_names_out()

def load_data(file1_name, file2_name):
    
    lst_lines = []
    # 1. load words from file1_name to a list
    f = open(file1_name, "r")
    line = f.readline().strip()
    while (line != ""):
        line = f.readline().strip()
        map = {}
        map[line] = 1
        lst_lines.append(map)
    f.close()
    
    # 2. load words from file2_name to the list
    f = open(file2_name, "r")
    line = f.readline().strip()
    while (line != ""):
        line = f.readline().strip()
        map = {}
        map[line] = 0
        lst_lines.append(map)
    f.close()
        
    # # 2. process all headlines from lst_lines to a tuple: (headline: line)
    # my_lst_of_map = []
    # for line in lst_lines:
    #     my_map = {}
    #     my_map["headline"] = line
    #     my_lst_of_map.append(my_map)
        
    # 3. randomly assign all words in lst_all_words into 3 groups
    n = len(lst_lines)
    k1 = round(n * 0.15)  #validation
    k2 = round(n * 0.15)  #test
    k3 = round(n * 0.7)   #trainning
    lst_validation = []
    lst_deletions_index = []
    i = 0
    while i < k1:
        random_index = random.randint(n)
        if random_index in lst_deletions_index:
            continue
        # my_lst_of_map[random_index]["headline"] = my_lst_of_map[random_index]["headline"][:-4]
        # lst_validation.append(my_lst_of_map[random_index])
        # lst_deletions_index.append(random_index)
        lst_validation.append(lst_lines[random_index])
        i = i+1
        # print("lst_validation: ", my_lst_of_map[random_index])
        # del my_lst_of_map[random_index]
    
    lst_test = []
    i = 0
    while i < k2:
        random_index = random.randint(n)
        if random_index in lst_deletions_index:
            continue
        # my_lst_of_map[random_index]["headline"] = my_lst_of_map[random_index]["headline"][:-4]
        # lst_test.append(my_lst_of_map[random_index])
        # print("lst_test ", my_lst_of_map[random_index])
        lst_test.append(lst_lines[random_index])
        lst_deletions_index.append(random_index)
        i = i+1
        
    lst_training = []
    labels = []
    i = 0
    while i < k3:
        random_index = random.randint(n)
        if random_index in lst_deletions_index:
            continue
        # line = my_lst_of_map[random_index]["headline"]
        # if line[-4:] == "real":
        #     labels.append(1)
        # elif line[-4:] == "fake":
        #     labels.append(0)
        # my_lst_of_map[random_index]["headline"] = my_lst_of_map[random_index]["headline"][:-4]
        lst_training.append(lst_lines[random_index])
        if list(lst_lines[random_index].values()).__eq__([0]):
            labels.append(0)
        else:
            labels.append(1)
        # print("lst_training: ", my_lst_of_map[random_index])
        lst_deletions_index.append(random_index)
        i = i + 1
    
    
    vec = DictVectorizer()
    X = vec.fit_transform(lst_training)   
    lst_features = vec.get_feature_names_out()
    print(lst_features)
    
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
    clf.fit(X, labels)
    print(clf.score(X, labels))
    
    
if __name__ == "__main__":
    filepath1 = Path(__file__).parent / "clean_real.txt"
    filepath2 = Path(__file__).parent / "clean_fake.txt"
    load_data(filepath1, filepath2)
     
     