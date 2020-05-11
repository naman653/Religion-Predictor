import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv('flag.csv')
colour_list = ['green', 'orange', 'red', 'brown', 'black', 'gold', 'grey', 'blue']
colour_list = sorted(colour_list)
landmass_list = ['N. America', 'S. America', 'Europe', 'Africa', 'Asia', 'Ocenia']
religion_list = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']
language_list = ['English', 'Spanish', 'French', 'German', 'Slavic', 'Other Indo-European', 'Chinese', 'Arabic',
                 'Altiac Family', 'Others']


# Swap categorical data and numerical values helper functions

def assign_colour(colour):
    return colour_list.index(colour)


def assign_landmass(number):
    return landmass_list[number - 1]


def assign_religion(number):
    return religion_list[number]


def assign_language(number):
    return language_list[number - 1]


# Change categorical values to numerical values
data['mainhue'] = data['mainhue'].apply(assign_colour)
data['topleft'] = data['topleft'].apply(assign_colour)
data['botright'] = data['botright'].apply(assign_colour)

# Empty value plot
sn.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Continent vs Religion Count Graph
ax = sn.countplot(x=data['landmass'].apply(assign_landmass), hue=data['religion'].apply(assign_religion))
ax.set(xlabel="Continents")
plt.show()

# Language vs Religion Count Graph
ax = sn.countplot(x=data['language'].apply(assign_language), hue=data['religion'].apply(assign_religion))
ax.set(xlabel="Language")
plt.show()

# Dominant Color vs Religion Count Graph
ax = sn.countplot(x=data['mainhue'], hue=data['religion'].apply(assign_religion))
ax.set(xlabel="Dominant Color")
plt.show()

# Bottom Left Color vs Religion Count Graph
ax = sn.countplot(x=data['botright'], hue=data['religion'].apply(assign_religion))
ax.set(xlabel="Bottom Left Color")
plt.show()

# Top Left Color vs Religion Count Graph
ax = sn.countplot(x=data['topleft'], hue=data['religion'].apply(assign_religion))
ax.set(xlabel="Top Left Color")
plt.show()

# Correlation Heatmap
ax = plt.subplot()
corr = data.corr()
sn.heatmap(corr, cmap="Blues_r", annot=True)
plt.show()

# Change categorical values to numerical values
data['mainhue'] = data['mainhue'].apply(assign_colour)
data['topleft'] = data['topleft'].apply(assign_colour)
data['botright'] = data['botright'].apply(assign_colour)

feature_names = ['mainhue', 'language', 'landmass', 'topleft', 'botright']
x = data[feature_names]
y = data['religion']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Decision Tree
clf = DecisionTreeClassifier().fit(x_train, y_train)
decision_tree_test_score = clf.score(x_test, y_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(decision_tree_test_score))

y_pred = clf.predict(x_test)
con = confusion_matrix(pd.DataFrame(y_test)['religion'].apply(assign_religion),
                       pd.DataFrame(y_pred)[0].apply(assign_religion),
                       labels=religion_list)
sn.heatmap(con, annot=True, xticklabels=religion_list, yticklabels=religion_list)
plt.title("Confusion Matrix for Decision Tree with accuracy = {:.2f}".format(decision_tree_test_score))
plt.show()

# K-NN
k_range = range(1, 25)
acc_test = []
for k in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    knn_test_score = knn.score(x_test, y_test)
    acc_test.append(knn_test_score)
plt.plot(k_range, acc_test)
plt.show()
knn_test_score = max(acc_test)
k = acc_test.index(knn_test_score)+1
print('Max Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knn_test_score))


knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
con = confusion_matrix(pd.DataFrame(y_test)['religion'].apply(assign_religion),
                       pd.DataFrame(y_pred)[0].apply(assign_religion),
                       labels=religion_list)
sn.heatmap(con, annot=True, xticklabels=religion_list, yticklabels=religion_list)
plt.title("Confusion Matrix for {}-Nearest Neighbour Tree with accuracy = {:.2f}".format(k, knn_test_score))
plt.show()

# Naive-Baye's
gnb = GaussianNB()
gnb.fit(x_train, y_train)
naive_bayes_test_score = gnb.score(x_test, y_test)
print('Accuracy of GNB classifier on test set: {:.2f}'
      .format(naive_bayes_test_score))

y_pred = gnb.predict(x_test)


con = confusion_matrix(pd.DataFrame(y_test)['religion'].apply(assign_religion),
                       pd.DataFrame(y_pred)[0].apply(assign_religion),
                       labels=religion_list)
sn.heatmap(con, annot=True, xticklabels=religion_list, yticklabels=religion_list)
plt.show()

# Comarison of accuracies for test set
accurracy_test = [decision_tree_test_score*100, knn_test_score*100, naive_bayes_test_score*100]
labels = ["Decision Tree", "KNN", "Naive Baye's"]
plt.bar(labels, accurracy_test, align='center')
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accurracy plot for Test Set of Different Classifiers")
plt.show()
