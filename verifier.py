"""
Vérification des calculs de analyses.py sur des cas dont on connaît la réponse.
Lancer : python verifier.py   (aucune connexion internet nécessaire)
"""
import numpy as np
import pandas as pd

import analyses as A

ok = True


def test(nom, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(("OK    " if condition else "ÉCHEC ") + nom + (f"  ({detail})" if detail else ""))


rng = np.random.default_rng(0)
idx = pd.bdate_range("2021-01-01", periods=1200)
m = pd.Series(rng.normal(0.0003, 0.01, 1200), index=idx)

# bootstrap : sans alpha, la p-valeur moyenne doit tourner autour de 0,5 ; avec un vrai alpha, elle doit être faible
ps = [A.bootstrap_alpha(pd.Series(0.8 * m.values + np.random.default_rng(s).normal(0, 0.006, 1200), index=idx).iloc[:300], m.iloc[:300], n_boot=300, graine=s)["p_boot"] for s in range(20)]
test("bootstrap sans talent : p moyenne ≈ 0,5", 0.35 < np.mean(ps) < 0.65, f"{np.mean(ps):.2f}")
y = pd.Series(0.8 * m.values + rng.normal(0.001, 0.006, 1200), index=idx)
test("bootstrap avec talent : p < 0,05", A.bootstrap_alpha(y.iloc[:300], m.iloc[:300], n_boot=500)["p_boot"] < 0.05)

# normalité
test("loi normale non rejetée sur des données normales", A.normalite(rng.normal(size=500))["jb_p"] > 0.01)
test("loi normale rejetée sur des queues épaisses (Student 3)", A.normalite(rng.standard_t(3, size=500))["jb_p"] < 0.01)

# FDR : personne de talent -> pi+ ≈ 0 ; 10 % de talents -> pi+ ≈ 0,10
d = pd.DataFrame({"performance": rng.normal(0, 0.03, 200), "risque": np.full(200, 0.03 * np.sqrt(252 / 20))})
test("FDR sans talent : pi+ ≈ 0", A.fdr_participants(d, 20)["pi_plus"] < 0.03)
d.loc[:19, "performance"] = 0.12
test("FDR 10 % de talents : pi+ ≈ 0,10", 0.06 < A.fdr_participants(d, 20)["pi_plus"] < 0.14)

# régression multifactorielle : retrouver les vrais bêtas
F = pd.DataFrame(rng.normal(0, 0.01, (900, 3)), index=idx[:900], columns=["a", "b", "c"])
r = A.regression_multi(0.8 * F["a"] + 0.3 * F["b"] - 0.2 * F["c"] + rng.normal(0, 0.004, 900), F)
test("bêtas retrouvés (0,8 / 0,3 / −0,2)", np.allclose(r["betas"].values, [0.8, 0.3, -0.2], atol=0.05), str(r["betas"].round(2).tolist()))
test("décomposition de variance : somme = 1", abs(A.decomposition_variance(r).sum() - 1) < 1e-9)

# nombre effectif de paris : 8 titres indépendants -> 8 ; 8 titres identiques -> 1
R = pd.DataFrame(rng.normal(0, 0.01, (2000, 8)))
w = pd.Series(1 / 8, index=R.columns)
test("paris indépendants : 8 titres indépendants ≈ 8", abs(A.nombre_effectif_paris(w, R)["enb"] - 8) < 0.5)
Rc = pd.DataFrame(np.repeat(rng.normal(0, 0.01, (2000, 1)), 8, axis=1) + rng.normal(0, 1e-4, (2000, 8)))
test("paris indépendants : 8 titres identiques ≈ 1", abs(A.nombre_effectif_paris(w, Rc)["enb"] - 1) < 0.1)

# turbulence : ≈ 1 en moyenne pour des rendements indépendants
test("turbulence moyenne ≈ 1", 0.9 < A.turbulence(pd.DataFrame(rng.normal(0, 0.01, (800, 8)), index=pd.bdate_range("2022-01-01", periods=800))).mean() < 1.15)

# Brinson : allocation + sélection + interaction = écart de performance
tb = pd.DataFrame({"w_p": [0.5, 0.5, 0], "r_p": [0.10, 0.02, np.nan], "w_b": [0.3, 0.3, 0.4], "r_b": [0.05, 0.03, -0.02]}, index=list("ABC"))
t, Rb = A.brinson(tb)
test("Brinson : la somme des effets = écart de performance", abs(t["total"].sum() - ((tb.w_p * tb.r_p.fillna(0)).sum() - Rb)) < 1e-12)

# parité de risque : contributions égales
C = np.cov(rng.normal(0, 0.01, (500, 6)).T) * 252
C[0, 0] *= 4
we = A.optimiser(C, "erc", 0.5)
rc = we * (C @ we) / (we @ C @ we)
test("parité de risque : contributions égales", np.allclose(rc, 1 / 6, atol=0.005), str(rc.round(3).tolist()))

# plafonnement des poids
wp = A.plafonner(rng.dirichlet(np.ones(25)), 0.06)
test("plafonnement : aucun poids au-dessus de 6 % et somme = 1", wp.max() <= 0.06 + 1e-9 and abs(wp.sum() - 1) < 1e-9)

# jumeaux : sans talent, percentile moyen ≈ 0,5
car = pd.DataFrame({"secteur": ["A"] * 60, "mom_q": [1] * 60, "vol_q": [1] * 60}, index=[f"T{i}" for i in range(60)])
pc = [A.jumeaux(car, pd.Series(0.1, index=car.index[:10]), pd.Series(np.random.default_rng(100 + s).normal(0, 0.05, 60), index=car.index), n=2000, graine=s)["percentile"] for s in range(30)]
test("jumeaux sans talent : percentile moyen ≈ 0,5", 0.4 < np.mean(pc) < 0.6, f"{np.mean(pc):.2f}")

# score de Brier
test("score de Brier d'un exemple connu = 0,14", abs(A.score_brier(pd.DataFrame({"probabilite": [0.8, 0.7, 0.6, 0.9, 0.3, 0.5], "resultat": [1, 1, 0, 1, 0, 1]}))["brier"] - 0.14) < 1e-9)

# lecture des fichiers de Kenneth French
txt = "Titre\n\n,Mkt-RF,SMB,HML\n20240102,  -0.52,   0.10,  0.32\n20240103,   0.20,  -0.12, 0.05\n\nCopyright\n"
ff = A.lire_french(txt)
test("lecture Fama-French : 2 jours, valeurs en décimal", len(ff) == 2 and abs(ff.iloc[0, 0] + 0.0052) < 1e-12)

print("\nTOUS LES TESTS PASSENT" if ok else "\nAU MOINS UN TEST ÉCHOUE")
