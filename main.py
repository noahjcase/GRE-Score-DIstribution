# Driver Script for GRE Score Examples
# Parameters
seed = 20250101
n_scores = 50000
tol_corr = 0.0025

# Import Libraries
with open('libraries.py') as file:
    code = file.read()
    exec(code)

# Set Seed
np.random.seed(seed)

# Load Custom Functions
spec = importlib.util.spec_from_file_location("corr_sim", "corr_sim.py")
corr_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corr_sim)


# Read in the `.csv` files and convert them to numpy arrays
aw_scores_df = pd.read_csv("aw-scores.csv")
vr_qr_scores_df = pd.read_csv("vr-qr-scores.csv")

# NB: We can use the same vector for VR and QR scores. Both are in the range
# [130, 170]. Percentiles are currently in the range [0,100), so divide by 100
vr_qr_scores = vr_qr_scores_df['Scaled Score'].to_numpy()

vr_percs = vr_qr_scores_df['Verbal Reasoning'].to_numpy()/100
qr_percs = vr_qr_scores_df['Quantitative Reasoning'].to_numpy()/100

aw_scores = aw_scores_df['Score Levels'].to_numpy()
aw_percs = aw_scores_df['Analytical Writing'].to_numpy()/100

# Convert percentiles to probabilities for the three sections and then
# sample from the marginal distributions.
vr_x, vr_prob = corr_sim.percentiles_to_probabilities(vr_qr_scores, vr_percs)
qr_x, qr_prob = corr_sim.percentiles_to_probabilities(vr_qr_scores, qr_percs)
aw_x, aw_prob = corr_sim.percentiles_to_probabilities(aw_scores, aw_percs)

vr_sample = np.random.choice(a=vr_x, size=n_scores, p=vr_prob)
qr_sample = np.random.choice(a=qr_x, size=n_scores, p=qr_prob)
aw_sample = np.random.choice(a=aw_x, size=n_scores, p=aw_prob)

# Now order the simulated score vectors per desired correlations
vr_ordered, qr_ordered = corr_sim.corr_sim(
    vr_sample, qr_sample, 0.35, tol_corr, 1000)
aw_ordered = corr_sim.multi_corr_sim(
    vr_ordered, qr_ordered, aw_sample, 0.63, 0.1, tol_corr, 20000)

simulated_joint_scores = pd.DataFrame(
    data=np.array([vr_ordered, qr_ordered,aw_ordered]).T,
    columns=['Verbal Reasoning','Quantitative Reasoning','Analytical Writing'])

simulated_joint_scores.to_csv("simulated_joint_scores.csv", index=False)
