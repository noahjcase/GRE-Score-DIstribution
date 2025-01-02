# Driver Script for GRE Score Examples
# Parameters
seed = 20250101
n_scores = 50000

# Import Libraries
with open('libraries.py') as file:
    code = file.read()
    exec(code)

# Load Custom Functions
spec = importlib.util.spec_from_file_location("corr_sim", "GRE Distribution.py")
corr_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corr_sim)


# Read in the `.csv` files and convert them to numpy arrays
aw_scores_df = pd.read_csv("aw-scores.csv")
vr_qr_scores_df = pd.read_csv("vr-qr-scores.csv")

# NB: We can use the same vector for VR and QR scores. Both are in the range
# [130, 170].
vr_qr_scores = vr_qr_scores_df['Scaled Score'].to_numpy()

vr_percs = vr_qr_scores_df['Verbal Reasoning'].to_numpy()
qr_percs = vr_qr_scores_df['Quantitative Reasoning'].to_numpy()

aw_scores = aw_scores_df['Score Levels'].to_numpy()
aw_percs = aw_scores_df['Analytical Writing'].to_numpy()

sampleScores = np.random.choice(scores, size=n_scores, p=probabilities)
