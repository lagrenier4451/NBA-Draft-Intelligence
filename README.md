# NBA Draft Analysis — PCA, Regression & Prospect Scoring

**Authors:** Luc-Alexandre Grenier & Patrick Reilly  
**Course:** Data Mining — Spring 2026  
**Dataset:** [wyattowalsh/basketball](https://www.kaggle.com/datasets/wyattowalsh/basketball) (Kaggle, version 231)

> This is Part 2 of our project. Part 1 (EDA) is at [bahaneku/nba-draft-analysis](https://github.com/bahaneku/nba-draft-analysis).

---

## What we looked at

We wanted to know whether the NBA combine actually predicts which players have long careers. The combine measures stuff like height, wingspan, vertical leap, sprint speed, and bench press — but does any of it matter?

We used three tables from the Kaggle dataset: combine measurements, draft history, and player career info. After joining them and filtering to 2001–2019 draft classes (to avoid penalizing recent picks who are still playing), we ended up with ~700 players for PCA and ~460 for regression.

Two main questions:
1. Do the combine measurements collapse into something interpretable?
2. Does any of it actually predict career length, or is it all just about where you get drafted?

---

## What we found

**PCA — the combine basically measures three things**

We ran PCA on 10 measurements and got 3 components that explained 75.6% of the variance:
- PC1 = Size (height, wingspan, standing reach, weight) — 48.5%
- PC2 = Explosiveness (vertical leap, sprint) — 16.5%
- PC3 = Strength (bench press, body fat) — 10.6%

Not surprising that size dominated — it's the most consistently measured thing at the combine.

**Regression — draft pick is almost everything**

We ran two models predicting career length:
- Model 1 (with draft pick): R² = 0.308
- Model 2 (combine only): R² = 0.113

Removing draft pick drops the R² by 0.196 — about 63.5% of Model 1's explanatory power comes from the pick alone. The combine factors were statistically insignificant in both models. Our interpretation: teams already use physical measurements when deciding where to pick someone, so once the pick is made, the combine doesn't add new information.

**Clustering — Quick Guards are undervalued**

K-Means with k=4 gave us four physical archetypes. The most interesting finding was that Quick Guards (small, explosive) had the highest rate of elite outcomes (2.78%) even though they get drafted latest on average. Curry, Paul, Lillard all fall in this cluster.

**Dark horses and busts**

We flagged late picks who outperformed expectations (Trevor Ariza, Isaiah Thomas, Marcin Gortat) and top picks who underperformed despite strong combine scores (Thabeet, Thomas Robinson). Kevin Durant had 0 bench press reps and below-average explosiveness — the combine completely missed him.

**2026 prospects**

We ran the 2026 combine class through our PCA model. Darryn Peterson's Quick Guard profile stood out — same archetype as the historically undervalued players.

| Prospect | Pick | Archetype | Pred. Career |
|---|---|---|---|
| AJ Dybantsa | #1 | Athletic Wing | 10.3 yrs |
| Darryn Peterson | #3 | Quick Guard | 10.2 yrs |
| Cameron Boozer | #5 | Athletic Wing | 9.7 yrs |
| Caleb Wilson | #9 | Athletic Wing | 9.1 yrs |
| Baba Miller | #17 | Athletic Wing | 7.9 yrs |
| Aday Mara | #22 | Athletic Wing | 6.9 yrs |
| Luigi Suigo | #28 | Athletic Wing | 6.2 yrs |

*Some athletic measurements were missing for 2026 prospects and were imputed at the historical mean.*

---

## How to run it

```bash
pip install -r requirements.txt
jupyter lab
```

Open `NBA.PCA.5.23.26.ipynb` and run the first cell first (it sets the working directory and loads the data). Then run the rest top to bottom.

The data CSVs are in `data/` and all plots save to `outputs/`.

---

## Limitations worth mentioning

- Career length isn't a perfect success measure — it captures longevity but not how good someone actually was
- Top prospects often skip combine testing, so the sample skews toward late-round players trying to prove themselves
- The 2026 scores use partial data — some athletic testing wasn't done yet
- A survival model would probably handle career length better than OLS since careers are censored
