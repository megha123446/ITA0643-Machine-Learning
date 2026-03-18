def find_s(training_data):
    # Initialize hypothesis
    hypothesis = ['∅'] * (len(training_data[0]) - 1)
    
    for example in training_data:
        if example[-1] == "Yes":  # Consider only positive examples
            for i in range(len(hypothesis)):
                if hypothesis[i] == '∅':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
    
    return hypothesis


# Training data
training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

print("Final Hypothesis:", find_s(training_data))