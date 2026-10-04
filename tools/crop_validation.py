"""Paper-1 style crop recommender (7 features, 22 crops) validated MORE strictly, with uncertainty.
Data: Kaggle 'Crop Recommendation Dataset' -> save as data/Crop_recommendation.csv (cols N,P,K,temperature,humidity,ph,rainfall,label)
Usage: python -m tools.crop_validation data/Crop_recommendation.csv
Reports: (1) random split + 10-fold CV (comparable to Paper 1), (2) climate-cluster hold-out (proxy for region shift),
(3) noise robustness, (4) split-conformal prediction sets: coverage under random vs shifted test data."""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
MODELS = {"RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
          "NaiveBayes": lambda: GaussianNB(),
          "Bagging": lambda: BaggingClassifier(random_state=0),
          "DecisionTree": lambda: DecisionTreeClassifier(random_state=0)}


def conformal_sets(model, Xcal, ycal, Xtest, alpha=0.1):
    """Split conformal (LAC score = 1 - p(true class)). Returns boolean matrix of prediction-set membership."""
    cls = list(model.classes_); idx = {c: i for i, c in enumerate(cls)}
    pc = model.predict_proba(Xcal)
    s = np.array([1 - pc[i, idx[y]] if y in idx else 1.0 for i, y in enumerate(ycal)])
    n = len(s); q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    P = model.predict_proba(Xtest)
    sets = P >= 1 - q
    sets[np.arange(len(P)), P.argmax(1)] = True      # never return an empty set (only raises coverage)
    return sets, cls


def coverage(sets, cls, ytest):
    idx = {c: i for i, c in enumerate(cls)}
    hit = [bool(sets[i, idx[y]]) if y in idx else False for i, y in enumerate(ytest)]
    return float(np.mean(hit)), float(sets.sum(1).mean())


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if valid.size else np.nan


def run(df, verbose=True):
    X, y = df[FEATS].to_numpy(dtype=float), df["label"].to_numpy()
    res = {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    for name, mk in MODELS.items():
        m = mk().fit(Xtr, ytr)
        cv = cross_val_score(mk(), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=0)).mean()
        res[name] = {"random_test_acc": m.score(Xte, yte), "cv10_acc": cv}
    # climate-cluster hold-out: whole climate zones unseen in training
    Z = StandardScaler().fit_transform(df[["temperature", "humidity", "rainfall"]])
    groups = KMeans(6, n_init=10, random_state=0).fit_predict(Z)
    for name, mk in MODELS.items():
        accs, unseen = [], []
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = mk().fit(X[tr], y[tr])
            seen = np.isin(y[te], m.classes_)
            unseen.append(1 - seen.mean())
            accs.append(m.score(X[te][seen], y[te][seen]) if seen.any() else np.nan)
        res[name].update({"climate_holdout_acc_on_seen_classes": safe_nanmean(accs),
                          "rows_with_class_unseen_in_train": float(np.mean(unseen))})
    # noise robustness (10% of feature std) for the best random-split model
    rf = MODELS["RandomForest"]().fit(Xtr, ytr)
    Xn = Xte + np.random.default_rng(0).normal(0, 0.1 * X.std(0), Xte.shape)
    res["RandomForest"]["noisy_test_acc"] = rf.score(Xn, yte)
    # conformal: coverage on random test vs climate-shifted test
    Xa, Xcal, ya, ycal = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=1)
    rf2 = MODELS["RandomForest"]().fit(Xa, ya)
    s, cls = conformal_sets(rf2, Xcal, ycal, Xte)
    res["RandomForest"]["conformal_random_cov_setsize"] = coverage(s, cls, yte)
    tr, te = next(GroupKFold(5).split(X, y, groups))
    Xa, Xcal, ya, ycal = train_test_split(X[tr], y[tr], test_size=0.25, random_state=1)
    rf3 = MODELS["RandomForest"]().fit(Xa, ya)
    s, cls = conformal_sets(rf3, Xcal, ycal, X[te])
    res["RandomForest"]["conformal_shifted_cov_setsize"] = coverage(s, cls, y[te])
    out = pd.DataFrame(res).T
    if verbose:
        print(out.round(3).to_string())
    return out


if __name__ == "__main__":
    run(pd.read_csv(sys.argv[1])).to_csv("data/crop_validation_results.csv")
