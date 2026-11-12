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

# Sharpe de Lo : identique au Sharpe simple sans autocorrélation, plus bas avec une autocorrélation positive
x = pd.Series(rng.normal(0.0005, 0.01, 3000))
s1 = A.sharpe_lo(x)
test("Sharpe de Lo = Sharpe simple sans autocorrélation", abs(s1["sr_lo"] - s1["sr_simple"]) < 0.05, f"{s1['sr_lo']:.3f} / {s1['sr_simple']:.3f}")
e = rng.normal(0, 0.01, 3000); ar = np.zeros(3000)
for i in range(1, 3000):
    ar[i] = 0.3 * ar[i - 1] + e[i]
s2 = A.sharpe_lo(pd.Series(ar + 0.0005))
test("Sharpe de Lo < Sharpe simple avec autocorrélation positive", s2["sr_lo"] < s2["sr_simple"] * 0.85, f"{s2['sr_lo']:.3f} / {s2['sr_simple']:.3f}")

# Sharpe dégonflé : plus d'essais = probabilité plus faible
test("Sharpe dégonflé : 50 essais < 1 essai", A.sharpe_degonfle(x.iloc[:250], 50)["dsr"] < A.sharpe_degonfle(x.iloc[:250], 1)["dsr"])

# VaR de Cornish-Fisher : ≈ VaR historique sur des données normales
test("VaR Cornish-Fisher ≈ historique (loi normale)", abs(A.var_cornish_fisher(x) + np.percentile(x, 5)) < 0.001)

# R² ajusté ≤ R²
rr = A.regression_multi(pd.Series(rng.normal(size=300)), pd.DataFrame(rng.normal(size=(300, 5))))
test("R² ajusté inférieur au R² brut", rr["r2_adj"] < rr["r2"])

# HHI : 25 lignes égales -> 0,04 et 25 lignes effectives
test("HHI de 25 lignes égales = 0,04", np.allclose(A.hhi(pd.Series(np.full(25, 0.04))), (0.04, 25.0)))

# ratios : un portefeuille identique à l'indice a un bêta de 1, un écart de suivi nul et des captures de 1
idx2 = pd.bdate_range("2024-01-01", periods=300)
b2 = pd.Series(rng.normal(0.0003, 0.01, 300), index=idx2)
tr, _ = A.ratios(b2, b2, 0.0)
v_ = tr.set_index("ratio")["portefeuille"]
test("ratios : copie de l'indice -> bêta 1, écart de suivi 0, capture 1", abs(v_["Bêta"] - 1) < 1e-9 and v_["Écart de suivi"] < 1e-12 and abs(v_["Capture haussière"] - 1) < 1e-9)

# v7 : qualité, investissement, révisions, dérive post-annonce, active share
test("rentabilité brute : 30 / 100 = 0,30 ; actif nul -> NaN", abs(A.rentabilite_brute(30, 100) - 0.3) < 1e-12 and np.isnan(A.rentabilite_brute(30, 0)))
test("croissance de l'actif : 110 / 100 -> +10 %", abs(A.variation_annuelle(110, 100) - 0.10) < 1e-12 and np.isnan(A.variation_annuelle(1, None)))
test("révision : de −2 à −1 = +50 % (valeur absolue)", abs(A.revision_prevision(-1, -2) - 0.5) < 1e-12)
s_, j_ = A.signal_pead(0.08, "2026-10-01", "2026-11-05"); s2_, _ = A.signal_pead(-0.08, "2026-06-01", "2026-11-05")
test("dérive post-annonce : surprise +8 % il y a 35 j -> +1 ; il y a 157 j -> 0", s_ == 1 and j_ == 35 and s2_ == 0)
test("active share : copie = 0, aucun titre commun = 1, moitié commune = 0,5",
     abs(A.active_share({"a": .5, "b": .5}, {"a": .5, "b": .5})) < 1e-12 and abs(A.active_share({"a": 1}, {"b": 1}) - 1) < 1e-12
     and abs(A.active_share({"a": .5, "b": .5}, {"a": .5, "c": .5}) - 0.5) < 1e-12)
test("active share : les poids sont ramenés à 100 % (cash ignoré)", abs(A.active_share({"a": .4, "b": .4}, {"a": 1, "b": 1})) < 1e-12)

# v8 : portefeuille modèle
_i = pd.bdate_range("2021-01-01", periods=900); _g = np.random.default_rng(42)
_derive = np.linspace(-0.0006, 0.0010, 30)                     # titres 0 (pire) à 29 (meilleur), dérive constante
_vols = np.tile([0.008, 0.012, 0.016], 10)
_px = pd.DataFrame(np.exp(np.cumsum(_g.normal(_derive, _vols, (900, 30)), axis=0)), index=_i, columns=[f"T{k:02d}" for k in range(30)])
_sec = {f"T{k:02d}": f"S{k % 6}" for k in range(30)}; _pla = {f"T{k:02d}": f"P{k % 4}" for k in range(30)}
_sig = A.signaux_modele(_px, _i[600])
_sig_fut = A.signaux_modele(_px.assign(T29=_px["T29"].where(_px.index <= _i[600], 999.0)), _i[600])
test("modèle : les signaux n'utilisent aucune donnée postérieure à la date", float((_sig - _sig_fut).abs().max().max()) < 1e-9 and len(_sig) == 30)
_tit, _sc = A.selection_modele(_sig, _sec, _pla, n=12, poids_signaux={"momentum": 1, "faible_vol": 0, "haut52": 0})
from collections import Counter
test("modèle : 12 titres, au plus 20 % par secteur et 30 % par place", len(_tit) == 12 and max(Counter(_sec[t] for t in _tit).values()) <= 2
     and max(Counter(_pla[t] for t in _tit).values()) <= 3)
test("modèle : le momentum retient des titres à dérive supérieure à la moyenne de l'univers", np.mean([int(t[1:]) for t in _tit]) > 14.5 + 1)
_w = A.poids_modele(_px, _i[600], _tit, _sec, _pla, mode="mdp", borne=0.10)
test("modèle : chaque ligne >= 1,5 %, somme 1, ligne <= 10 %, secteur <= 20 %", abs(_w.sum() - 1) < 1e-6 and _w.min() >= 0.015 - 1e-6 and _w.max() <= 0.10 + 1e-6
     and max(_w.groupby([_sec[t] for t in _w.index]).sum()) <= 0.20 + 1e-4)
_bt = A.backtest_modele(_px, _sec, _pla, n=12, poids_signaux={"momentum": 1, "faible_vol": 0, "haut52": 0}, mode="egal")
test("modèle : sur des titres à dérive persistante, le momentum bat l'univers à poids égaux hors échantillon",
     _bt is not None and (1 + _bt["r"]).prod() > (1 + _bt["r_egal"]).prod() and 0 <= _bt["rotation"] <= 1)
_sg = A.singes_modele(_px, _sec, _pla, list(_bt["compositions"].keys()), n=12, n_singes=200)
_cf = A.confirmations_modele(_bt["r"], _bt["r_egal"], _px.mean(axis=1).pct_change(), _sg, n_essais=5, n_boot=200)
test("modèle : les singes couvrent les mêmes jours et le modèle les bat (p < 5 %)", _sg.shape == (len(_bt["r"]), 200) and _cf["p_rendement"] < 0.05)
_pxh = pd.DataFrame(np.exp(np.cumsum(_g.normal(0.0002, 0.012, (900, 30)), axis=0)), index=_i, columns=_px.columns)   # aucun signal : pur hasard
_bth = A.backtest_modele(_pxh, _sec, _pla, n=12, mode="egal")
_cfh = A.confirmations_modele(_bth["r"], _bth["r_egal"], _pxh.mean(axis=1).pct_change(), A.singes_modele(_pxh, _sec, _pla, list(_bth["compositions"].keys()), n=12, n_singes=200), n_boot=200)
test("modèle : sur des cours purement aléatoires, le test ne crie pas au talent (p > 5 %)", _cfh["p_rendement"] > 0.05 and _cfh["p_sharpe"] > 0.05)
_b0 = pd.Series(_g.normal(0, 0.01, 2000)); _p0 = np.where(_b0 > 0, 1.3 * _b0, 0.6 * _b0) + _g.normal(0, 0.001, 2000)
_rt, _ = A.ratios(pd.Series(_p0), _b0)
_rt = _rt.set_index("ratio")["portefeuille"]
test("bêta à la hausse 1,3 et bêta à la baisse 0,6 retrouvés", abs(_rt["Bêta à la hausse"] - 1.3) < 0.05 and abs(_rt["Bêta à la baisse"] - 0.6) < 0.05)

_sx = pd.Series([100, 110, 120, 115, 100], index=pd.bdate_range("2026-10-01", periods=5))
_st = A.stop_suiveur(_sx, "2026-10-01", 0.24)       # seuil = 2,5 × 24 % / √12 = 17,3 %
test("stop suiveur : recul de 16,7 % depuis le plus haut, seuil 17,3 %, pas encore touché", abs(_st["recul"] + 1/6) < 1e-9 and abs(_st["seuil"] + 0.1732) < 1e-3 and not _st["touche"])
test("stop suiveur : même recul avec 20 % de volatilité (seuil 14,4 %) -> touché ; seuil borné à 8 % et 30 %",
     A.stop_suiveur(_sx, "2026-10-01", 0.20)["touche"] and abs(A.stop_suiveur(_sx, None, 0.05)["seuil"] + 0.08) < 1e-9 and abs(A.stop_suiveur(_sx, None, 0.90)["seuil"] + 0.30) < 1e-9)

_ia = pd.bdate_range("2024-01-01", periods=400); _ga = np.random.default_rng(5)
_ma = _ga.normal(0.0003, 0.01, 400)
_bh = np.array([1.4, 1.0, 0.7, 1.0, 1.2, 0.9, 1.1, 0.8]); _bb = np.array([0.6, 1.0, 1.3, 1.0, 0.8, 1.1, 0.9, 1.2])   # la moyenne de l'univers garde un bêta de 1
_ra = np.where(_ma[:, None] > 0, _ma[:, None] * _bh, _ma[:, None] * _bb) + _ga.normal(0, 0.002, (400, 8))
_pa = pd.DataFrame(np.exp(np.cumsum(_ra, axis=0)), index=_ia, columns=list("ABCDEFGH"))
_as = A.signaux_modele(_pa, _ia[-1])["asym_beta"]
test("asymétrie de bêta : A (1,4 / 0,6) la plus forte, C (0,7 / 1,3) la plus faible, valeurs proches de +0,8 et −0,6",
     _as.idxmax() == "A" and _as.idxmin() == "C" and abs(_as["A"] - 0.8) < 0.15 and abs(_as["C"] + 0.6) < 0.15)
_ce = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]); _we = A.optimiser(_ce, "erc", 1.0); _rc = _we * (_ce @ _we)
test("parité de risque (gradient exact) : les trois contributions au risque sont égales", np.ptp(_rc) / _rc.mean() < 1e-3 and abs(_we.sum() - 1) < 1e-9)

# v9 : scénarios politiques
_ip = pd.bdate_range("2024-01-01", periods=600); _gp = np.random.default_rng(9)
_mk = pd.Series(_gp.normal(0.0003, 0.009, 600), _ip); _fa = pd.Series(_gp.normal(0, 0.004, 600), _ip)
_Rp = pd.DataFrame({"FR": 1.0 * _mk + 1.5 * _fa + _gp.normal(0, 0.003, 600), "EU": 0.8 * _mk + 0.0 * _fa + _gp.normal(0, 0.003, 600)}, index=_ip)
_sf = A.sensibilite_facteur(_Rp, _mk, _fa)
test("sensibilité au facteur France : 1,5 retrouvé pour le titre exposé (t élevé), 0 pour l'autre",
     abs(_sf.loc["FR", "beta_facteur"] - 1.5) < 0.1 and _sf.loc["FR", "t_facteur"] > 10 and abs(_sf.loc["EU", "beta_facteur"]) < 0.1 and abs(_sf.loc["EU", "beta_marche"] - 0.8) < 0.05)
_pb = A.prix_binaire(0.30, -0.10, 1.0)
test("événement à deux issues : p = 30 %, écart −10 % -> −7 % si réalisé, +3 % si écarté, −1 % pour 10 points de plus ; espérance nulle",
     abs(_pb["realise"] + 0.07) < 1e-12 and abs(_pb["ecarte"] - 0.03) < 1e-12 and abs(_pb["dix_points"] + 0.01) < 1e-12
     and abs(0.30 * _pb["realise"] + 0.70 * _pb["ecarte"]) < 1e-12)
_pr = pd.Series(np.clip(0.3 + np.cumsum(_gp.normal(0, 0.01, 600)), 0.02, 0.98), _ip)
_fr = -0.08 * _pr.diff().fillna(0) + pd.Series(_gp.normal(0, 0.0005, 600), _ip)
_rc = A.reaction_aux_cotes(_fr, _pr); _rc0 = A.reaction_aux_cotes(pd.Series(_gp.normal(0, 0.004, 600), _ip), _pr)
test("réaction aux cotes : écart implicite de −8 % retrouvé, et aucun lien détecté sur un facteur indépendant",
     abs(_rc["pente"] + 0.08) < 0.01 and abs(_rc["t"]) > 5 and abs(_rc0["t"]) < 3)

# --- v9.1 : cours avec trous (nuit, jours fériés) et signal alpha
_gq = np.random.default_rng(5); _iq = pd.bdate_range("2023-01-02", periods=700)
_mq = _gq.normal(0.0003, 0.01, 700)
_Rq = {f"T{i}": (0.5 + 0.1 * i) * _mq + _gq.normal(0, 0.008, 700) for i in range(15)}
_Rq["T3"] = _Rq["T3"] + 0.0015            # alpha pur : aucun bêta en plus
_Rq["T14"] = 2.5 * _mq + _gq.normal(0, 0.008, 700)   # monte par son bêta seul
_Pq = pd.DataFrame({k: 50 * np.exp(np.cumsum(v)) for k, v in _Rq.items()}, index=_iq)
_sq = A.signaux_modele(_Pq, _iq[-1])
test("signal alpha : le titre à alpha pur est premier, le titre à fort bêta sans alpha n'est pas dans le haut du classement",
     _sq["alpha"].idxmax() == "T3" and _sq["alpha"].rank(ascending=False)["T14"] > 3)
_Pt = _Pq.copy()
_Pt.loc[_iq[-1] + pd.Timedelta(days=1)] = np.nan          # ligne de nuit : seules les devises cotent
_Pt.iloc[-23, :5] = np.nan; _Pt.iloc[-254, 5:9] = np.nan  # jours fériés locaux aux dates exactes utilisées par les signaux
_st = A.signaux_modele(_Pt, _Pt.index[-1])
test("cours avec trous : la ligne vide de la nuit et les jours fériés locaux n'éliminent aucun titre",
     len(_st) == 15 and _st.notna().all().all() and abs(_st["momentum"] - _sq["momentum"]).max() < 0.05)
_b1 = A.backtest_modele(_Pq, {c: "S" for c in _Pq}, {c: "P" for c in _Pq}, 6, None, "equal", 0.2, 1.0, 1.0)
_b2 = A.backtest_modele(_Pt, {c: "S" for c in _Pq}, {c: "P" for c in _Pq}, 6, None, "equal", 0.2, 1.0, 1.0)
test("test hors échantillon : les trous ne font pas perdre de rendement (écart cumulé inférieur à 1 point)",
     _b1 is not None and _b2 is not None and abs((1 + _b1["r"]).prod() - (1 + _b2["r"]).prod()) < 0.01)

# --- v11 : scénarios
_rb = A.robustesse_modele(_Pq, _iq[-1], {c: "S" for c in _Pq}, {c: "P" for c in _Pq}, 6, {"momentum": 0, "faible_vol": 0, "alpha": 1}, "equal", 0.2,
                          n_scen=60, cap_secteur=1.0, cap_place=1.0)
test("solidité : le titre à alpha planté est retenu dans presque tous les scénarios, les fréquences restent entre 0 et 1 et somment à 6",
     _rb["frequence"]["T3"] > 0.9 and _rb["frequence"].between(0, 1).all() and abs(_rb["frequence"].sum() - 6) < 1e-9 and abs(_rb["poids_moyens"].sum() - 1) < 1e-6)
_sf2, _ = A.selection_modele(_sq, {c: "S" for c in _Pq}, {c: "P" for c in _Pq}, 6, {"momentum": 1, "faible_vol": 0}, 1.0, 1.0, forces=[_sq["momentum"].idxmin()])
test("titre imposé : il est retenu même avec le plus mauvais score, et le portefeuille garde 6 lignes", _sq["momentum"].idxmin() in _sf2 and len(_sf2) == 6)
_gs = np.random.default_rng(9); _is = pd.bdate_range("2023-01-02", periods=750)
_ms = _gs.normal(0.0004, 0.010, 750)
_Rs = pd.DataFrame({"calme": 0.5 * _ms + _gs.normal(0, 0.002, 750), "agité": 1.5 * _ms + _gs.normal(0, 0.004, 750), "indice": _ms}, index=_is)
_t1, _c1 = A.simulation_horizon(_Rs, 120, n_sim=4000, centrer_sur="indice")
_th = 0.010 * np.sqrt(120)
test("simulation : dispersion à 120 jours conforme à la théorie (σ × √120), baisse en cours de route toujours négative ou nulle",
     abs(_t1["indice"].std() / _th - 1) < 0.12 and (_c1.values <= 1e-12).all() and _t1["agité"].std() > 2 * _t1["calme"].std())
test("simulation « aucun avantage » : les trois portefeuilles ont le même rendement médian à 1,5 point près",
     abs(_t1["calme"].median() - _t1["indice"].median()) < 0.015 and abs(_t1["agité"].median() - _t1["indice"].median()) < 0.02)
_t2, _ = A.simulation_horizon(_Rs, 120, n_sim=4000)
test("simulation : même graine, mêmes résultats (reproductible)", np.allclose(_t2.values, A.simulation_horizon(_Rs, 120, n_sim=4000)[0].values))

print("\nTOUS LES TESTS PASSENT" if ok else "\nAU MOINS UN TEST ÉCHOUE")
