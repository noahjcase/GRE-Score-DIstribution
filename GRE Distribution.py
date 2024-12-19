"""Various Utilities for understanding the distribution of GRE Scores"""

# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    sampleScores = nnp.random.choice(scores, size=nScores, p=probabilities)
    return sampleScores

nScores = 100000
awSample = SampleScores(awScoresDf,
                        "Score Levels", "Analytical Writing", nScores=nScores)
vrSample = SampleScores(vrQrScoresDf,
                        "Scaled Score", "Verbal Reasoning", nScores=nScores)
qrSample = SampleScores(vrQrScoresDf,
                        "Scaled Score", "Quantitative Reasoning", nScores=nScores)

sections = [awSample, vrSample, qrSample]

# Quick check that we've gotten the standard deviation and means right
# We ought to have. The percentiles determine the distribution!

for s in sections:
    print(np.mean(s))
    print(np.std(s))

# Function that takes as an input two vectors and a target correlation

def corrScoreVec(sortVec, toOrderVec, targetCorr, tolCorr, maxIter):
    """
    Function to return a new version of usingVec that is shuffled so that its
    correlation with sortedVec is approximately equal to targetCorrr
    """

    masterVec = np.sort(sortVec.copy())
    usingVec = np.shuffle(toOrderVec.copy())

    corr = np.corrcoeef(masterVec, usingVec)[0,1]
    iter = 0
    while abs(corr - targetCorr) >= tolCorr and iter <= maxIter :
        if corr < targetCorr:
            
        else:

        


        corr = np.corrcoeef(masterVec, usingVec)[0,1]
        iter += 1