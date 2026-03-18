import csv
import copy

def load_csv(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
    return data

def initialize_S(num_attributes):
    return ['0'] * num_attributes  # Most specific hypothesis

def initialize_G(num_attributes):
    return [['?'] * num_attributes]  # Most general hypothesis

def is_more_general(h1, h2):
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = (x == '?' or (x != '0' and (x == y)))
        more_general_parts.append(mg)
    return all(more_general_parts)

def min_generalizations(h, x):
    new_h = h.copy()
    for i in range(len(h)):
        if h[i] == '0':
            new_h[i] = x[i]
        elif h[i] != x[i]:
            new_h[i] = '?'
    return new_h

def min_specializations(h, domains, x):
    specializations = []
    for i in range(len(h)):
        if h[i] == '?':
            for value in domains[i]:
                if x[i] != value:
                    new_h = h.copy()
                    new_h[i] = value
                    specializations.append(new_h)
        elif h[i] != '0':
            new_h = h.copy()
            new_h[i] = '0'
            specializations.append(new_h)
    return specializations

def candidate_elimination(data):
    num_attributes = len(data[0]) - 1
    domains = [set() for _ in range(num_attributes)]
    
    for row in data[1:]:
        for i in range(num_attributes):
            domains[i].add(row[i])

    S = initialize_S(num_attributes)
    G = initialize_G(num_attributes)

    for row in data[1:]:
        x = row[:-1]
        label = row[-1]

        if label == "Yes":
            G = [g for g in G if is_more_general(g, x)]

            if not is_more_general(S, x):
                S = min_generalizations(S, x)

            G = [g for g in G if is_more_general(g, S)]

        else:  # Negative example
            G_new = []
            for g in G:
                if is_more_general(g, x):
                    specializations = min_specializations(g, domains, x)
                    G_new.extend(specializations)
                else:
                    G_new.append(g)

            G = [g for g in G_new if is_more_general(g, S)]

    return S, G


# ----------------------------
# Demonstration
# ----------------------------
data = load_csv("training_data.csv")
S_final, G_final = candidate_elimination(data)

print("Final Specific Hypothesis (S):")
print(S_final)

print("\nFinal General Hypotheses (G):")
for g in G_final:
    print(g)