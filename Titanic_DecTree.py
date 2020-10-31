import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn

df = pd.read_csv('train.csv')
y = df['Survived']
df.drop('Survived', axis = 1, inplace = True)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace = True)
# Заменили отсутствующие возраста медианным значением
df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)

categorical = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
df = pd.concat([df, pd.get_dummies(df[categorical], columns=categorical, drop_first=True)],axis=1)
df.drop(categorical, axis=1, inplace=True)

# Разбиваем на трейн-тест
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.1, random_state = 13)
# Добавляем в трейн целевую величину для обучения
X_train['target'] = y_train

print('Train size: {0}, Test size: {1}'.format(len(X_train), len(X_test)))

'''
Вершина дерева - словарь вида
{ 'condition' : <...> - условие, по которому проверяем
  'threshold' : <...> - порог условия
  'left' : левое поддерево - такой же словарь (вершина) / метка класса
  'right': правое поддерево - словарь / метка класса
}
'''
def dominant_class(dataframe):
    classes = (dataframe['target']).value_counts()
    return classes.index[0]

def Shennon_entropy(series): # Энтропия шеннона
    classes = sorted(series.value_counts(), reverse = True)
    total_count = sum(classes)
    probabilities = list(map(lambda count: count/total_count, classes))
    entropy = sum(list(map(lambda p: (-p)*math.log2(p), probabilities)))
    return entropy

def feature_separation(dataframe, feature_name):
    sorted_dataframe = dataframe.sort_values(by = [feature_name], ignore_index = True)[['target', feature_name]]
    primary_entropy = Shennon_entropy(sorted_dataframe['target'])
    threshold = None
    max_info_gain = 0

    for i in range(1, len(sorted_dataframe['target']) - 1):
        left_part = sorted_dataframe[:i]
        right_part = sorted_dataframe[i:]
        left_entropy = Shennon_entropy(left_part['target'])
        right_entropy = Shennon_entropy(right_part['target'])
        info_gain = primary_entropy - (len(left_part)/len(sorted_dataframe))*left_entropy - (len(right_part)/len(sorted_dataframe))*right_entropy
        if (info_gain > max_info_gain):
            max_info_gain = info_gain
            threshold = (sorted_dataframe[feature_name][i - 1] + sorted_dataframe[feature_name][i])/2
    return (max_info_gain, threshold)

def data_separation(dataframe):
    optimal_feature, threshold, childs = None, None, dataframe
    total_features = [f for f in dataframe.columns.tolist() if f != 'target']
    max_info_gain = 0

    for feature in total_features:
        separation = feature_separation(dataframe, feature)
        if (separation[0] > max_info_gain):
            max_info_gain = separation[0]
            optimal_feature = feature
            threshold = separation[1]
    left_child = dataframe[dataframe[optimal_feature] <= threshold]
    right_child = dataframe[dataframe[optimal_feature] > threshold]
    return {'condition' : optimal_feature, 'threshold' : threshold, 'childs' : (left_child, right_child)}

def build_tree(node, max_depth, curr_depth):
    ch_left, ch_right = node['childs']
    del(node['childs'])
    if (len(ch_left) == 0 or len(ch_right) == 0):
        node['left'] = node['right'] = dominant_class(ch_left) if len(ch_right) == 0 else dominant_class(ch_right)
        return
    if (curr_depth >= max_depth):
        node['left'] = dominant_class(ch_left)
        node['right'] = dominant_class(ch_right)
        return
    else:
        node['left'] = data_separation(ch_left)
        build_tree(node['left'], max_depth, curr_depth + 1)
        node['right'] = data_separation(ch_right)
        build_tree(node['right'], max_depth, curr_depth + 1)
        
def fit_tree(dataframe, max_depth):
    root_node = data_separation(dataframe)
    build_tree(root_node, max_depth, 1)
    return root_node

def object_prediction(data_row, tree): # Предсказание для одного обьекта
    if (data_row[tree['condition']] <= tree['threshold']):
        if (type(tree['left']) == dict):
            return object_prediction(data_row, tree['left'])
        else:
            return tree['left']
    else:
        if (type(tree['right']) == dict):
            return object_prediction(data_row, tree['right'])
        else:
            return tree['right']

def data_prediction(dataframe, tree): # Предсказание для набора обьектов
    prediction = pd.Series()
    for index, row in dataframe.iterrows():
        result = pd.Series(object_prediction(row, tree), [index])
        prediction = prediction.append(result)
    return prediction

tree = fit_tree(X_train, 3)

predict_test = data_prediction(X_test, tree)
predict_train = data_prediction(X_train, tree)

# Сравнение с sklearn - деревом
skl_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 3)

del X_train['target']
skl_tree.fit(X_train, y_train)

skl_test = skl_tree.predict(X_test)
skl_train = skl_tree.predict(X_train)

skl_roc_train = roc_auc_score(y_train, skl_train)
skl_roc_test = roc_auc_score(y_test, skl_test)

my_roc_train = roc_auc_score(y_train, predict_train)
my_roc_test = roc_auc_score(y_test, predict_test)

print(skl_roc_train)
print(skl_roc_test)
print('----------')
print(my_roc_train)
print(my_roc_test)
print('----------')