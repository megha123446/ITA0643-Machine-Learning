import pandas as pd
import math

data_dict = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "PlayTennis": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
}

data = pd.DataFrame(data_dict)

def entropy(target_col):
    counts = target_col.value_counts()
    total = len(target_col)
    ent = 0
    for count in counts:
        p = count / total
        ent -= p * math.log2(p)
    return ent

def information_gain(data, attribute, target="PlayTennis"):
    total_entropy = entropy(data[target])
    values = data[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def id3(data, original_data, features, target="PlayTennis"):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    if len(features) == 0:
        return original_data[target].mode()[0]

    gains = [information_gain(data, feature, target) for feature in features]
    best_feature = features[gains.index(max(gains))]
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, original_data, remaining_features, target)
        tree[best_feature][value] = subtree

    return tree

features = list(data.columns[:-1])
decision_tree = id3(data, data, features)

print("Decision Tree:")
print(decision_tree)

def classify(sample, tree):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    value = sample[root]
    subtree = tree[root][value]
    return classify(sample, subtree)

new_sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

prediction = classify(new_sample, decision_tree)

print("\nNew Sample:", new_sample)
print("Prediction:", prediction)