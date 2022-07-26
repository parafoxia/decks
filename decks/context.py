from math import dist

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

# ANGER:    (-.51,  .59,  .25)
# FEAR:     (-.64,  .60, -.43)
# JOY:      ( .76,  .48,  .35)
# LOVE:     ( .82,  .65, -.05)
# SADNESS:  (-.63, -.27, -.33)
# SURPRISE: ( .40,  .67, -.13)

points = np.array(
    [
        [-.51,  .59,  .25],
        [-.64,  .60, -.43],
        [ .76,  .48,  .35],
        [ .82,  .65, -.05],
        [-.63, -.27, -.33],
        [ .40,  .67, -.13],
    ]
)
max_d = max([dist(p, q) for p in points for q in points])


def deca(preds):
    progress = np.empty((len(preds), 3))

    def get_midpoint(pred):
        return np.sum(points * pred[:, None], axis=0) / np.sum(pred)

    def shift(a, b):
        dist = a - b
        move = dist * .2
        return a - move

    dp = get_midpoint(preds[0])
    progress[0] = dp

    for i, pred in enumerate(preds[1:], start=1):
        mp = get_midpoint(pred)
        dp = shift(dp, mp)
        progress[i] = dp

    norm_d = np.array([dist(dp, p) for p in points]) / max_d
    return 1 - norm_d, progress


if __name__ == "__main__":
    preds = np.array(
        [[0.9, 0.05, 0.0, 0.0, 0.05, 0.0],
         [0.0, 0.0 , 0.8, 0.2, 0.0 , 0.0],
         [0.0, 0.0 , 0.8, 0.2, 0.0 , 0.0],
         [0.0, 0.0 , 0.8, 0.2, 0.0 , 0.0],
         [0.0, 0.0 , 0.8, 0.2, 0.0 , 0.0]]
    )

    confidence, progress = deca(preds)
    print(confidence)

    fig = plt.figure(figsize=(10, 10))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    sns.scatterplot(x=points[:, 0], y=points[:, 1])
    sns.lineplot(x=progress[:, 0], y=progress[:, 1])
    sns.scatterplot(x=[progress[-1][0]], y=[progress[-1][1]])
    fig.savefig("test.png")
