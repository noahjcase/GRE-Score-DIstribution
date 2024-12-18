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
    sampleScores = np.random.choice(scores, size=nScores, p=probabilities)
    return sampleScores

nScores = 100000
awSample = SampleScores(awScoresDf,
                        "Score Levels", "Analytical Writing", nScores=nScores)
vrSample = SampleScores(vrQrScoresDf,
                        "Scaled Score", "Verbal Reasoning", nScores=nScores)
qrSample = SampleScores(vrQrScoresDf,
                        "Scaled Score", "Quantitative Reasoning", nScores=nScores)

sections = [awSample, vrSample, qrSample]
for s in sections:
    print(np.mean(s))
    print(np.std(s))
