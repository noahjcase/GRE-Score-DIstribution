# Driver Script for GRE Score Examples
# Parameters
seed = 19980103
n_scores = 50000

# Import Libraries
with open('libraries.py') as file:
    code = file.read()
    exec(code)

sampleScores = np.random.choice(scores, size=nScores, p=probabilities)
