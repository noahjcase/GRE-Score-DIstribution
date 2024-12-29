"""Various Utilities for understanding the distribution of GRE Scores"""

# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Read the csv files to pd data frams
awScoresDf = pd.read_csv("aw-scores.csv")
vrQrScoresDf = pd.read_csv("vr-qr-scores.csv")

# function to generate some number of of scores satisfying a given distribution
def SampleScores(df, scoreCol, percCol, nScores):
    """
    Generate a vector of scores matching a given distribution.

    Parameters:
        df (pd.DataFrame): DataFrame containing scores and their percentiles.
        score_col (str): Column name for the score levels.
        freq_col (str): Column name for the percentiles.
        nScores (int): Number of scores to generate.

    Returns:
        np.ndarray: A vector of scores sampled according to the distribution.
            implies by the percentiles.
    """
    if scoreCol not in df.columns or percCol not in df.columns:
        raise ValueError(f"Columns '{scoreCol}' and '{percCol}' must exist in the DataFrame '{df}'.")

    validDf = df.copy()
    validDf[percCol] = validDf[percCol].fillna(0)
    
     # Define the scores and percentiles

    scores = validDf[scoreCol].to_numpy()
    percentiles = validDf[percCol].to_numpy()
    
    sorted_indices = np.argsort(-percentiles)  # Negative for descending order
    scores = scores[sorted_indices]
    percentiles = percentiles[sorted_indices]

    # replace the percentiles with probabilities
    probabilities = percentiles.copy()
    for i, p in enumerate(percentiles):
        if i == 0:
            probabilities[i] = (100 - percentiles[i])/100
            continue
        else:
            probabilities[i] = (percentiles[i - 1] - percentiles[i])/100
    
    assert np.isclose(np.sum(probabilities), 1)
    sampleScores = np.random.choice(scores, size=nScores, p=probabilities)
    return sampleScores

nScores = 100
awSample = SampleScores(awScoresDf,
                        "Score Levels", "Analytical Writing", nScores=nScores)
vrSample = SampleScores(vrQrScoresDf,
                        "Scaled Score", "Verbal Reasoning", nScores=nScores)
qrSample = SampleScores(vrQrScoresDf,
                        "Scaled Score", "Quantitative Reasoning", nScores=nScores)

sections = [awSample, vrSample, qrSample]

# Quick check that we've gotten the standard deviation and means right.
# We ought to have. The percentiles determine the distribution!

for s in sections:
    print(np.mean(s))
    print(np.std(s))

# Function that takes as an input two vectors and a target correlation

def corrScoreVec(sortVec, toOrderVec, targetCorr, tolCorr, maxIter):
    """
    Function to adjust toOrderVec so its correlation with sortVec 
    is approximately equal to targetCorr, with a chance of overshooting.
    """
    assert len(sortVec) == len(toOrderVec)

    nScores = len(sortVec)
    masterVec = np.sort(sortVec)  # Sorting ensures consistency
    usingVec = toOrderVec.copy()
    np.random.shuffle(usingVec)

    corr = np.corrcoef(masterVec, usingVec)[0, 1]
    print(f"Initial correlation: {corr}")

    iter = 0
    swaps = 0
    while abs(corr - targetCorr) >= tolCorr and iter < maxIter:
        # Dynamic number of swaps based on distance from target correlation
        numSwaps = max(math.floor(nScores/100), math.ceil(nScores * (abs(corr - targetCorr) / 10)))

        for _ in range(numSwaps):
            # Randomly select two indices
            idx1, idx2 = np.random.choice(nScores, 2, replace=False)

            # Ensure idx1 < idx2 for consistent logic
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1

            # Get values
            master1, master2 = masterVec[idx1], masterVec[idx2]
            use1, use2 = usingVec[idx1], usingVec[idx2]

            if corr < targetCorr:
                # Normal adjustment: Make swaps to move toward target correlation
                if (master1 < master2 and use1 > use2) or (master1 > master2 and use1 < use2):
                    usingVec[idx1], usingVec[idx2] = use2, use1
            else:
                # Reverse adjustment: Introduce overshooting
                if (master1 < master2 and use1 < use2) or (master1 > master2 and use1 > use2):
                    usingVec[idx1], usingVec[idx2] = use2, use1
        # Update correlation
        corr = np.corrcoef(masterVec, usingVec)[0, 1]
        iter += 1
        swaps += numSwaps

        # Optional: Print progress periodically
        if iter % 50 == 0:
            print(f"Iteration {iter}, Swaps {swaps}, correlation: {corr}")

    print(f"Final correlation: {corr} after {iter} iterations and {swaps} swaps")
    return masterVec, usingVec
qrVec, vrVec = corrScoreVec(qrSample, vrSample, 0.35, 0.005, 10000)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(qrVec, vrVec, marker = "+")
ax.set_xlim([130, 170])
ax.set_ylim([130, 170])
ax.set_xlabel("Quantitative Reasoning")
ax.set_ylabel("Verbal Reasoning")
plt.show()

np.corrcoef(qrVec, vrVec)