"""
Calculs de l'appli NextWise, sans Streamlit (testables seuls).

Références principales :
- Monte Carlo / portefeuilles aléatoires : Tarnavskyi (2020), Burns (2004)
- Bootstrap de l'alpha : Kosowski, Timmermann, Wermers et White (2006) ; Fama et French (2010)
- Bootstrap par blocs stationnaires : Politis et Romano (1994)
- Portefeuilles jumeaux (sélectivité par caractéristiques) : Daniel, Grinblatt, Titman et Wermers (1997) ; Heyman et al. (2014)
- Sharpe probabiliste : Bailey et López de Prado (2012)
- Taux de fausses découvertes : Barras, Scaillet et Wermers (2010) ; Storey (2002)
- R² et sélectivité : Amihud et Goyenko (2013)
- Momentum : Jegadeesh et Titman (1993) ; Novy-Marx (2012) ; Blitz, Huij et Martens (2011) ;
  Da, Gurun et Warachka (2014) ; George et Hwang (2004) ; Daniel et Moskowitz (2016) ; Barroso et Santa-Clara (2015)
- Tendance : Faber (2007) ; Moskowitz, Ooi et Pedersen (2012)
- Régime : Kritzman et Li (2010) ; Kritzman, Li, Page et Rigobon (2011) ; Choueifaty et Coignard (2008)
"""
import numpy as np
import pandas as pd
from scipy import stats


# ======================================================================= utilitaires
def rendements(prix):
    return prix.pct_change(fill_method=None)


def mensuels(prix):
    """Rendements mensuels à partir de prix quotidiens (dernier cours de chaque mois)."""
    m = prix.resample("ME").last()
    return m.pct_change(fill_method=None)


def regression(y, x):
    """MCO simple y = a + b x + e. Retourne a, b, résidus, t de a, R²."""
    z = pd.concat([y, x], axis=1).dropna()
    if len(z) < 10:
        return None
    yv, xv = z.iloc[:, 0].values, z.iloc[:, 1].values
    X = np.column_stack([np.ones(len(xv)), xv])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ coef
    n, k = len(yv), 2
    s2 = res @ res / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    t_a = coef[0] / np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.nan
    r2 = 1 - (res @ res) / ((yv - yv.mean()) @ (yv - yv.mean())) if yv.var() > 0 else np.nan
    return {"alpha": coef[0], "beta": coef[1], "residus": pd.Series(res, index=z.index), "t_alpha": t_a,
            "r2": r2, "n": n, "x": pd.Series(xv, index=z.index), "y": pd.Series(yv, index=z.index)}


# ======================================================================= chance ou talent
def normalite(r):
    """Tests de normalité des rendements (Jarque-Bera, Shapiro-Wilk) + asymétrie et kurtosis."""
    x = pd.Series(r).dropna().values
    if len(x) < 8:
        return None
    jb = stats.jarque_bera(x)
    sw = stats.shapiro(x[-5000:])
    return {"n": len(x), "asymetrie": float(stats.skew(x)), "kurtosis": float(stats.kurtosis(x, fisher=False)),
            "jb_p": float(jb.pvalue), "sw_p": float(sw.pvalue)}


def blocs_stationnaires(n, longueur_moy, rng):
    """Indices d'un tirage bootstrap par blocs stationnaires (Politis et Romano, 1994)."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / max(longueur_moy, 1)
    i = rng.integers(n)
    for k in range(n):
        if k == 0 or rng.random() < p:
            i = rng.integers(n)
        else:
            i = (i + 1) % n
        idx[k] = i
    return idx


def bootstrap_alpha(r_ptf, r_marche, n_boot=2000, longueur_bloc=5, graine=1):
    """Bootstrap de Kosowski et al. (2006) : on construit un monde sans talent (alpha = 0) avec les vrais
    résidus et le vrai marché, rééchantillonnés par blocs, et on regarde combien de fois le t de l'alpha
    y est au moins aussi élevé que le tien."""
    reg = regression(r_ptf, r_marche)
    if reg is None:
        return None
    rng = np.random.default_rng(graine)
    x, e = reg["x"].values, reg["residus"].values
    n = len(x)
    t_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = blocs_stationnaires(n, longueur_bloc, rng)
        yb = reg["beta"] * x[idx] + e[idx]          # alpha = 0 par construction
        X = np.column_stack([np.ones(n), x[idx]])
        coef, *_ = np.linalg.lstsq(X, yb, rcond=None)
        res = yb - X @ coef
        s2 = res @ res / (n - 2)
        v = s2 * np.linalg.inv(X.T @ X)[0, 0]
        t_boot[b] = coef[0] / np.sqrt(v) if v > 0 else 0.0
    p_boot = float((t_boot >= reg["t_alpha"]).mean())
    p_param = float(1 - stats.t.cdf(reg["t_alpha"], df=n - 2))
    return {**{k: reg[k] for k in ("alpha", "beta", "t_alpha", "r2", "n")}, "p_boot": p_boot, "p_param": p_param,
            "t_boot": t_boot}


def sharpe_probabiliste(r, sr_ref=0.0):
    """PSR de Bailey et López de Prado (2012) : probabilité que le vrai Sharpe (par période) dépasse sr_ref,
    en tenant compte de la longueur, de l'asymétrie et de la kurtosis des rendements."""
    x = pd.Series(r).dropna().values
    n = len(x)
    if n < 10 or x.std(ddof=1) == 0:
        return None
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        return None
    z = (sr - sr_ref) * np.sqrt(n - 1) / np.sqrt(denom)
    z_normal = (sr - sr_ref) * np.sqrt(n - 1) / np.sqrt(1 + sr ** 2 / 2)  # même calcul en supposant la loi normale
    return {"sr": sr, "sr_annuel": sr * np.sqrt(252), "psr": float(stats.norm.cdf(z)),
            "psr_normal": float(stats.norm.cdf(z_normal)), "n": n, "asymetrie": g3, "kurtosis": g4}


def tableau_survie(W, prix_univ, prix_bench, w_moi, debut, fin):
    """Tarnavskyi (2020) appliqué mois par mois : part des portefeuilles aléatoires qui battent l'indice
    sur chaque mois, et si toi tu le bats."""
    p = prix_univ.loc[debut:fin].ffill()
    b = prix_bench.loc[debut:fin].ffill()
    if len(p) < 2:
        return pd.DataFrame()
    bornes = list(p.resample("ME").last().index)
    dates = [p.index[0]] + [p.index[p.index <= d][-1] for d in bornes]
    dates = sorted(set(dates))
    lignes = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        r = (p.loc[d1] / p.loc[d0] - 1).fillna(0).values
        rb = float(b.loc[d1] / b.loc[d0] - 1)
        ra = W @ r
        rm = float(w_moi @ r)
        lignes.append({"periode": f"{d0:%d/%m} → {d1:%d/%m/%Y}", "indice": rb, "part_alea_bat_indice": float((ra > rb).mean()),
                       "toi": rm, "tu_bats_indice": rm > rb, "ton_percentile": float((ra < rm).mean())})
    return pd.DataFrame(lignes)


def jumeaux(caract, w_moi, rend, n=10_000, graine=3):
    """Portefeuilles jumeaux (Daniel et al., 1997 ; Heyman et al., 2014) : chaque ligne est remplacée par une
    action tirée au hasard ayant les mêmes caractéristiques (secteur, tiers de momentum, tiers de volatilité).
    caract : DataFrame indexé par ticker (secteur, mom_q, vol_q) sur tout l'univers + tes lignes ;
    w_moi : Series ticker -> poids ; rend : Series ticker -> rendement sur la période."""
    rng = np.random.default_rng(graine)
    tick = list(caract.index)
    pos = {t: i for i, t in enumerate(tick)}
    R = rend.reindex(tick).values
    choix, niveaux = [], []
    for t, w in w_moi.items():
        if t not in caract.index or w <= 0:
            continue
        c = caract.loc[t]
        cand = caract[(caract["secteur"] == c["secteur"]) & (caract["mom_q"] == c["mom_q"]) & (caract["vol_q"] == c["vol_q"])]
        niveau = "secteur + momentum + vol"
        if len(cand) < 3:
            cand = caract[(caract["mom_q"] == c["mom_q"]) & (caract["vol_q"] == c["vol_q"])]
            niveau = "momentum + vol"
        if len(cand) < 3:
            cand = caract[caract["vol_q"] == c["vol_q"]]
            niveau = "vol"
        idx = np.array([pos[x] for x in cand.index if np.isfinite(R[pos[x]])])
        if len(idx) == 0:
            continue
        choix.append((w, idx))
        niveaux.append({"ticker": t, "jumeaux_possibles": len(idx), "critere": niveau})
    if not choix:
        return None
    r_jum = np.zeros(n)
    r_moi = 0.0
    for (w, idx), info in zip(choix, niveaux):
        tirage = idx[rng.integers(len(idx), size=n)]
        r_jum += w * R[tirage]
        r_moi += w * R[pos[info["ticker"]]]
    return {"r_jumeaux": r_jum, "r_moi": float(r_moi), "percentile": float((r_jum < r_moi).mean()),
            "p": float((r_jum >= r_moi).mean()), "details": pd.DataFrame(niveaux)}


def fdr_participants(df, jours, perf_ref=0.0, lam=0.5, gamma=0.10):
    """Barras, Scaillet et Wermers (2010) sur les participants : df a les colonnes performance (période, en
    décimal) et risque (volatilité annualisée, décimal). Le t de chaque participant compare son excès de
    performance à ce que son risque rend plausible sur la période."""
    d = df.dropna(subset=["performance", "risque"]).copy()
    d = d[d["risque"] > 0]
    if len(d) < 20:
        return None
    sigma_periode = d["risque"] * np.sqrt(max(jours, 1) / 252)
    d["t"] = (d["performance"] - perf_ref) / sigma_periode
    d["p"] = 2 * (1 - stats.norm.cdf(np.abs(d["t"])))
    N = len(d)
    pi0 = min(1.0, float((d["p"] > lam).sum() / (N * (1 - lam))))
    sig_pos = int(((d["p"] <= gamma) & (d["t"] > 0)).sum())
    sig_neg = int(((d["p"] <= gamma) & (d["t"] < 0)).sum())
    faux = pi0 * N * gamma / 2                          # faux positifs attendus de chaque côté
    pi_plus = max(0.0, (sig_pos - faux) / N)
    pi_moins = max(0.0, (sig_neg - faux) / N)
    fdr_plus = min(1.0, faux / sig_pos) if sig_pos else np.nan
    return {"N": N, "pi0": pi0, "pi_plus": pi_plus, "pi_moins": pi_moins, "sig_pos": sig_pos, "sig_neg": sig_neg,
            "faux_attendus": faux, "fdr_plus": fdr_plus, "table": d}


# ======================================================================= momentum
def signaux_momentum(prix, bench):
    """Signaux par titre (prix quotidiens en euros, ajustés).
    mom_12_1 : Jegadeesh et Titman ; mom_12_7 : Novy-Marx (2012) ; mom_resid : Blitz, Huij et Martens (2011)
    (somme des résidus mensuels t-12..t-2 face au marché, divisée par leur écart-type, régression sur 36 mois) ;
    id_fip : Da, Gurun et Warachka (2014), plus bas = tendance plus régulière ; haut52 : George et Hwang (2004) ;
    rend_1m : pour l'alerte d'inversion à 1 mois (Jegadeesh, 1990) ; tendance_10m : cours / moyenne 210 jours (Faber, 2007)."""
    rq = rendements(prix)
    rm = mensuels(prix)
    rbm = mensuels(bench.to_frame("b"))["b"]
    out = {}
    for t in prix.columns:
        s = prix[t].dropna()
        if len(s) < 260:
            out[t] = {}
            continue
        mom121 = s.iloc[-22] / s.iloc[-253] - 1
        mom127 = s.iloc[-148] / s.iloc[-253] - 1 if len(s) > 253 else np.nan
        r = rq[t].dropna().iloc[-252:-21]
        id_fip = np.sign(mom121) * ((r < 0).mean() - (r > 0).mean())
        haut = s.iloc[-1] / s.iloc[-252:].max()
        r1m = s.iloc[-1] / s.iloc[-22] - 1
        sma = s.iloc[-210:].mean()
        tend = s.iloc[-1] / sma - 1
        resid = np.nan
        y = rm[t].dropna().iloc[-37:-1]
        reg = regression(y, rbm) if len(y) >= 24 else None
        if reg is not None:
            e = reg["residus"]
            e12 = e.iloc[-12:-1] if len(e) >= 12 else e
            resid = e12.sum() / e12.std(ddof=1) if e12.std(ddof=1) > 0 else np.nan
        out[t] = {"mom_12_1": mom121, "mom_12_7": mom127, "mom_resid": resid, "id_fip": id_fip,
                  "haut52": haut, "rend_1m": r1m, "tendance_10m": tend}
    df = pd.DataFrame(out).T
    # score composite : moyenne des z-scores (le FIP est inversé : plus régulier = mieux)
    z = pd.DataFrame({c: (df[c] - df[c].mean()) / df[c].std(ddof=0) for c in ["mom_resid", "mom_12_7", "haut52"] if c in df})
    if "id_fip" in df:
        z["fip"] = -(df["id_fip"] - df["id_fip"].mean()) / df["id_fip"].std(ddof=0)
    df["score"] = z.mean(axis=1)
    return df


def facteur_momentum(prix, frequence_mois=1):
    """Facteur momentum européen maison (gagnants − perdants) sur l'univers : chaque mois, achat du tiers
    au meilleur momentum 12-1, vente du tiers au plus faible, à poids égaux. Retourne les rendements quotidiens."""
    rq = rendements(prix)
    fins = prix.resample("ME").last().index
    serie = []
    for d0, d1 in zip(fins[:-1], fins[1:]):
        hist = prix.loc[:d0]
        if len(hist) < 260:
            continue
        mom = hist.iloc[-22] / hist.iloc[-253] - 1
        mom = mom.dropna()
        if len(mom) < 6:
            continue
        q = mom.quantile([1 / 3, 2 / 3])
        g, p = mom[mom >= q.iloc[1]].index, mom[mom <= q.iloc[0]].index
        fen = rq.loc[(rq.index > d0) & (rq.index <= d1)]
        serie.append(fen[g].mean(axis=1) - fen[p].mean(axis=1))
    if not serie:
        return pd.Series(dtype=float)
    return pd.concat(serie).dropna()


def risque_krach_momentum(bench, wml):
    """Daniel et Moskowitz (2016) : le momentum s'effondre surtout quand le marché a baissé sur 2 ans et que la
    volatilité est forte. Barroso et Santa-Clara (2015) : on réduit l'exposition quand la vol du momentum monte."""
    b = bench.dropna()
    marche_2ans = b.iloc[-1] / b.iloc[-505] - 1 if len(b) > 505 else np.nan
    rb = b.pct_change().dropna()
    vol_marche = rb.iloc[-126:].std() * np.sqrt(252)
    seuil_vol = (rb.rolling(126).std() * np.sqrt(252)).dropna().quantile(0.75) if len(rb) > 300 else np.nan
    vol_wml = wml.iloc[-126:].std() * np.sqrt(252) if len(wml) > 60 else np.nan
    echelle = min(2.0, 0.12 / vol_wml) if vol_wml and vol_wml > 0 else np.nan
    marche_baissier = bool(pd.notna(marche_2ans) and marche_2ans < 0)
    vol_forte = bool(pd.notna(seuil_vol) and vol_marche > seuil_vol)
    return {"marche_2ans": marche_2ans, "vol_marche": vol_marche, "seuil_vol": seuil_vol, "vol_wml": vol_wml,
            "echelle_bsc": echelle, "marche_baissier": marche_baissier, "vol_forte": vol_forte,
            "alerte": marche_baissier and vol_forte}


# ======================================================================= régime de marché
def turbulence(R, fenetre=252):
    """Indice de turbulence de Kritzman et Li (2010) : distance de Mahalanobis des rendements du jour par rapport
    à la moyenne et aux covariances de l'année précédente. Grand = journée anormale au vu des corrélations."""
    R = R.dropna(how="any")
    vals, idx = [], []
    for i in range(fenetre, len(R)):
        h = R.iloc[i - fenetre:i]
        mu, cov = h.mean().values, np.cov(h.values, rowvar=False)
        inv = np.linalg.pinv(cov)
        d = R.iloc[i].values - mu
        vals.append(float(d @ inv @ d) / R.shape[1])
        idx.append(R.index[i])
    return pd.Series(vals, index=idx)


def ratio_absorption(R, fenetre=252, part=0.2, pas=5):
    """Ratio d'absorption de Kritzman et al. (2011) : part de la variance totale expliquée par le premier
    cinquième des facteurs principaux. Élevé = marchés soudés, donc fragiles."""
    R = R.dropna(how="any")
    k = max(1, int(round(R.shape[1] * part)))
    vals, idx = [], []
    for i in range(fenetre, len(R), pas):
        h = R.iloc[i - fenetre:i]
        ev = np.sort(np.linalg.eigvalsh(np.cov(h.values, rowvar=False)))[::-1]
        vals.append(float(ev[:k].sum() / ev.sum()))
        idx.append(R.index[i])
    return pd.Series(vals, index=idx)


def nombre_effectif_paris(poids, R):
    """Nombre de paris réellement indépendants = carré du ratio de diversification
    (Choueifaty et Coignard, 2008 ; Choueifaty, Froidure et Reynier, 2013). 25 lignes parfaitement
    indépendantes et de même risque donnent 25 ; 25 lignes qui bougent toutes ensemble donnent 1."""
    t = [x for x in poids.index if x in R.columns]
    if len(t) < 2:
        return None
    w = poids.reindex(t).fillna(0).values
    cov = R[t].dropna(how="all").fillna(0).cov().values
    var = float(w @ cov @ w)
    if var <= 0:
        return None
    dr = float((w * np.sqrt(np.diag(cov))).sum() / np.sqrt(var))
    val = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return {"enb": dr ** 2, "dr": dr, "n_lignes": len(t), "part_premier_facteur": float(val.max() / val.sum())}
