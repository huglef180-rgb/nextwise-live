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
    t_b = coef[1] / np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.nan
    r2 = 1 - (res @ res) / ((yv - yv.mean()) @ (yv - yv.mean())) if yv.var() > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k) if pd.notna(r2) and n > k else np.nan   # k = 2 paramètres (constante + bêta)
    return {"alpha": coef[0], "beta": coef[1], "residus": pd.Series(res, index=z.index), "t_alpha": t_a, "t_beta": t_b,
            "r2": r2, "r2_adj": r2_adj, "n": n, "x": pd.Series(xv, index=z.index), "y": pd.Series(yv, index=z.index)}


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


# ======================================================================= facteurs
def regression_multi(y, X):
    """MCO y = a + X b + e avec plusieurs facteurs. Retourne alpha, bêtas, t, R² et résidus."""
    z = pd.concat([pd.Series(y, name="__y"), X], axis=1).dropna()
    if len(z) < X.shape[1] + 15:
        return None
    yv = z["__y"].values
    Xv = np.column_stack([np.ones(len(z)), z[X.columns].values])
    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    res = yv - Xv @ coef
    n, k = Xv.shape
    s2 = res @ res / (n - k)
    cov = s2 * np.linalg.pinv(Xv.T @ Xv)
    t = coef / np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    r2 = 1 - (res @ res) / ((yv - yv.mean()) @ (yv - yv.mean())) if yv.var() > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k) if pd.notna(r2) and n > k else np.nan   # k = constante + facteurs
    return {"alpha": coef[0], "t_alpha": t[0], "betas": pd.Series(coef[1:], index=X.columns),
            "t": pd.Series(t[1:], index=X.columns), "r2": r2, "r2_adj": r2_adj, "n": n, "residus": pd.Series(res, index=z.index),
            "X": z[X.columns]}


def decomposition_variance(reg):
    """Part de la variance du portefeuille expliquée par chaque facteur (b_i × (F b)_i) et part propre (résidus)."""
    F = reg["X"].cov().values
    b = reg["betas"].values
    contrib = b * (F @ b)
    var_res = reg["residus"].var()
    tot = contrib.sum() + var_res
    out = pd.Series(np.append(contrib, var_res) / tot, index=list(reg["betas"].index) + ["Propre aux titres"])
    return out


def facteur_basse_vol(prix):
    """Facteur « basse volatilité » maison (dans l'esprit de Frazzini et Pedersen, Betting Against Beta, 2014) :
    chaque mois, achat du tiers le moins volatil sur un an, vente du tiers le plus volatil, à poids égaux."""
    rq = rendements(prix)
    fins = prix.resample("ME").last().index
    serie = []
    for d0, d1 in zip(fins[:-1], fins[1:]):
        h = rq.loc[:d0].iloc[-252:]
        if len(h) < 200:
            continue
        v = h.std().dropna()
        v = v[h.count() > 150]
        if len(v) < 6:
            continue
        q = v.quantile([1 / 3, 2 / 3])
        bas, haut = v[v <= q.iloc[0]].index, v[v >= q.iloc[1]].index
        fen = rq.loc[(rq.index > d0) & (rq.index <= d1)]
        serie.append(fen[bas].mean(axis=1) - fen[haut].mean(axis=1))
    return pd.concat(serie).dropna() if serie else pd.Series(dtype=float)


def lire_french(texte):
    """Lit un fichier CSV de la bibliothèque de Kenneth French (données quotidiennes) : lignes de titre, une ligne
    d'en-tête qui commence par une virgule, puis des lignes AAAAMMJJ,valeurs en %. S'arrête à la fin du bloc."""
    lignes = texte.replace("\r", "").split("\n")
    entete, data, dedans = None, [], False
    for l in lignes:
        if not dedans:
            if l.startswith(",") and any(c.isalpha() for c in l):
                entete = [c.strip() for c in l.split(",")[1:]]
                dedans = True
            continue
        p = [c.strip() for c in l.split(",")]
        if len(p) < 2 or not (p[0].isdigit() and len(p[0]) == 8):
            if data:
                break
            continue
        try:
            data.append([pd.Timestamp(p[0])] + [float(x) / 100 for x in p[1:len(entete) + 1]])
        except ValueError:
            break
    if not entete or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["date"] + entete).set_index("date")
    return df


def brinson(table):
    """Attribution de Brinson-Fachler par secteur. table : index secteur, colonnes w_p, r_p, w_b, r_b.
    Allocation : avoir sur- ou sous-pondéré les bons secteurs. Sélection : avoir choisi de meilleurs titres dans le secteur."""
    t = table.copy()
    t["r_p"] = t["r_p"].where(t["w_p"] > 0, t["r_b"])
    Rb = (t["w_b"] * t["r_b"]).sum()
    t["allocation"] = (t["w_p"] - t["w_b"]) * (t["r_b"] - Rb)
    t["selection"] = t["w_b"] * (t["r_p"] - t["r_b"])
    t["interaction"] = (t["w_p"] - t["w_b"]) * (t["r_p"] - t["r_b"])
    t["total"] = t["allocation"] + t["selection"] + t["interaction"]
    return t, Rb


def sensibilites(R, facteurs):
    """Bêtas de chaque titre à plusieurs facteurs (régression quotidienne)."""
    out = {}
    for c in R.columns:
        reg = regression_multi(R[c], facteurs)
        if reg is not None:
            out[c] = reg["betas"]
    return pd.DataFrame(out).T


# ======================================================================= optimisation
def _contraintes_poids(n, borne, plancher=0.0):
    from scipy.optimize import Bounds
    return Bounds(np.full(n, min(plancher, 1 / n)), np.full(n, borne)), [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]


def optimiser(cov, mode, borne=0.10, groupes=None, plancher=0.0, maxiter=500, ftol=1e-12):
    """Portefeuilles fondés sur le risque seul (sans prévision de rendement, donc robustes) :
    'minvar' variance minimale, 'mdp' diversification maximale (Choueifaty et Coignard, 2008),
    'erc' parité de risque (Maillard, Roncalli et Teïletche, 2010).
    groupes : liste de (masque booléen, plafond) pour limiter le poids d'un secteur ou d'une place."""
    from scipy.optimize import minimize
    n = cov.shape[0]
    sig = np.sqrt(np.diag(cov))
    w0 = np.full(n, 1 / n)
    bornes, cons = _contraintes_poids(n, max(borne, 1 / n + 1e-6), plancher)
    for masque, plafond in (groupes or []):
        m = np.asarray(masque, dtype=bool)
        if m.any() and m.sum() / n <= plafond + 1e-9:      # ignorée si les poids égaux la violent déjà (point de départ infaisable)
            cons = cons + [{"type": "ineq", "fun": lambda w, m=m, c=plafond: c - w[m].sum(), "jac": lambda w, m=m: -m.astype(float)}]
    jac = None          # gradients exacts : dix fois plus rapide que la dérivation numérique
    if mode == "minvar":
        f = lambda w: w @ cov @ w * 1e4
        jac = lambda w: 2e4 * (cov @ w)
    elif mode == "mdp":
        f = lambda w: -(w @ sig) / np.sqrt(w @ cov @ w)
        def jac(w):
            v = w @ cov @ w
            return -(sig / np.sqrt(v) - (w @ sig) * (cov @ w) / v ** 1.5)
    elif mode == "erc":
        # écart des parts de risque à 1/n ; formulation sans unité : l'ancienne (× 1e8) pouvait rester bloquée sur les poids égaux
        def f(w):
            g = cov @ w
            return ((w * g / (w @ g) - 1 / n) ** 2).sum() * 1e2
        def jac(w):
            g = cov @ w
            v = w @ g
            sh = w * g / v
            d = sh - 1 / n
            return 2e2 / v * (d * g + cov @ (w * d) - 2 * g * (d @ sh))
    else:
        return w0
    r = minimize(f, w0, jac=jac, method="SLSQP", bounds=bornes, constraints=cons, options={"maxiter": maxiter, "ftol": ftol})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def plafonner(w, borne, iterations=50):
    """Ramène chaque poids sous la borne en redistribuant l'excédent sur les autres lignes, proportionnellement."""
    w = w.copy()
    for _ in range(iterations):
        trop = w > borne + 1e-12
        if not trop.any():
            break
        exces = (w[trop] - borne).sum()
        w[trop] = borne
        libres = ~trop & (w < borne)
        if not libres.any():
            break
        w[libres] += exces * w[libres] / w[libres].sum()
    return w


def nuage_portefeuilles(cov, mu, n=4000, borne=0.10, graine=5):
    """Portefeuilles aléatoires (poids de Dirichlet de concentration variable, plafonnés) : volatilité, rendement passé,
    ratio de diversification, Sharpe."""
    rng = np.random.default_rng(graine)
    k = cov.shape[0]
    borne = max(borne, 1.0 / k + 1e-6)
    alphas = rng.uniform(0.3, 6.0, size=n)
    W = np.vstack([plafonner(rng.dirichlet(np.full(k, a_)), borne) for a_ in alphas])
    sig = np.sqrt(np.diag(cov))
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    ret = W @ mu
    return pd.DataFrame({"vol": vol, "rendement": ret, "dr": (W @ sig) / vol, "sharpe": ret / vol})


# ======================================================================= signaux dans le temps
def ic_mensuel(prix, n_mois=36):
    """Coefficient d'information mensuel (corrélation de rang entre un signal et le rendement du mois suivant,
    sur tout l'univers). Grinold et Kahn (2000) : un IC moyen de 0,05 est déjà un bon signal."""
    fins = list(prix.resample("ME").last().index)
    out = []
    for d0, d1 in zip(fins[-n_mois - 1:-1], fins[-n_mois:]):
        h = prix.loc[:d0]
        if len(h) < 260:
            continue
        p0 = h.iloc[-1]
        sig = pd.DataFrame({
            "Momentum 12-1": h.iloc[-22] / h.iloc[-253] - 1,
            "Momentum 12-7": h.iloc[-148] / h.iloc[-253] - 1,
            "Plus haut 52 semaines": p0 / h.iloc[-252:].max(),
            "Dernier mois (inversion)": -(p0 / h.iloc[-22] - 1),
        })
        p1 = prix.loc[:d1].iloc[-1]
        fut = p1 / p0 - 1
        ligne = {"mois": d1}
        for c in sig.columns:
            z = pd.concat([sig[c], fut], axis=1).dropna()
            ligne[c] = stats.spearmanr(z.iloc[:, 0], z.iloc[:, 1])[0] if len(z) > 10 else np.nan
        out.append(ligne)
    return pd.DataFrame(out).set_index("mois") if out else pd.DataFrame()


def scores_historiques(prix, bench, n_semaines=26):
    """Score de momentum de chaque titre recalculé chaque semaine sur les n dernières semaines."""
    dates = prix.index[::-5][:n_semaines][::-1]
    out = {}
    for d in dates:
        sg = signaux_momentum(prix.loc[:d], bench.loc[:d])
        out[d] = sg["score"]
    return pd.DataFrame(out)


# ======================================================================= prévisions
def score_brier(df):
    """Score de Brier (Brier, 1950) des prévisions résolues : moyenne de (probabilité − résultat)². 0 = parfait,
    0,25 = toujours répondre 50 %. Avec la calibration par tranches de probabilité."""
    d = df.dropna(subset=["probabilite", "resultat"]).copy()
    if d.empty:
        return None
    p, o = d["probabilite"].astype(float), d["resultat"].astype(float)
    bs = float(((p - o) ** 2).mean())
    base = float(o.mean())
    bs_ref = float(((base - o) ** 2).mean())
    d["tranche"] = pd.cut(p, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    cal = d.groupby("tranche", observed=True).agg(prevu=("probabilite", "mean"), realise=("resultat", "mean"), n=("resultat", "size"))
    return {"brier": bs, "n": len(d), "taux_reussite": base, "brier_reference": bs_ref,
            "skill": 1 - bs / bs_ref if bs_ref > 0 else np.nan, "calibration": cal}


# ======================================================================= ratios de performance (logiciels pros)
def sharpe_lo(r, rf_annuel=0.0, lags=10):
    """Sharpe annualisé avec taux sans risque et correction d'autocorrélation de Lo (2002) :
    SR_an = SR_jour × q / √(q + 2 Σ (q − k) ρ_k), q = 252 jours. Sans autocorrélation, on retrouve SR_jour × √252."""
    x = pd.Series(r).dropna()
    if len(x) < 20 or x.std(ddof=1) == 0:
        return {"sr_simple": np.nan, "sr_lo": np.nan}
    ex = x - rf_annuel / 252
    sr = ex.mean() / ex.std(ddof=1)
    q = 252
    L = min(lags, len(x) // 4)
    rho = [ex.autocorr(k) for k in range(1, L + 1)]
    rho = [p for p in rho if pd.notna(p)]
    denom = q + 2 * sum((q - k) * p for k, p in enumerate(rho, start=1))
    sr_lo = sr * q / np.sqrt(denom) if denom > 0 else np.nan
    return {"sr_simple": sr * np.sqrt(252), "sr_lo": sr_lo, "rho1": rho[0] if rho else np.nan}


def sharpe_degonfle(r, n_essais, rf_annuel=0.0):
    """Sharpe « dégonflé » (Bailey et López de Prado, 2014) : probabilité que le vrai Sharpe soit positif une fois
    pris en compte le fait d'avoir essayé n_essais versions et gardé la meilleure. La variance des Sharpe des essais est
    approchée par celle de l'estimateur, (1 + SR²/2) / (n − 1)."""
    x = pd.Series(r).dropna() - rf_annuel / 252
    n = len(x)
    if n < 20 or x.std(ddof=1) == 0:
        return None
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    v_sr = (1 + sr ** 2 / 2) / (n - 1)
    gamma = 0.5772156649
    N = max(int(n_essais), 1)
    sr0 = np.sqrt(v_sr) * ((1 - gamma) * stats.norm.ppf(1 - 1 / N) + gamma * stats.norm.ppf(1 - 1 / (N * np.e))) if N > 1 else 0.0
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        return None
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(denom)
    return {"dsr": float(stats.norm.cdf(z)), "sr_seuil_annuel": sr0 * np.sqrt(252), "sr_annuel": sr * np.sqrt(252), "n_essais": N}


def var_cornish_fisher(r, niveau=0.95):
    """VaR de Cornish-Fisher (Favre et Galeano, 2002) : la VaR normale corrigée de l'asymétrie et de la kurtosis."""
    x = pd.Series(r).dropna()
    if len(x) < 20:
        return np.nan
    z = stats.norm.ppf(1 - niveau)
    s, k = stats.skew(x), stats.kurtosis(x)          # kurtosis en excès
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(-(x.mean() + zcf * x.std(ddof=1)))


def ratios(r, rb, rf_annuel=0.0):
    """Tableau de ratios façon Morningstar / Bloomberg pour une série de rendements quotidiens r face à l'indice rb."""
    z = pd.concat([pd.Series(r, name="p"), pd.Series(rb, name="b")], axis=1).dropna()
    if len(z) < 20:
        return None
    p, b = z["p"], z["b"]
    rf = rf_annuel / 252
    n = len(z)
    an = lambda x: (1 + x).prod() ** (252 / n) - 1
    vol = lambda x: x.std(ddof=1) * np.sqrt(252)
    def dd(x):
        v = (1 + x).cumprod()
        return v / v.cummax() - 1
    def sortino(x):
        neg = np.minimum(x - rf, 0)
        dside = np.sqrt((neg ** 2).mean()) * np.sqrt(252)
        return (an(x) - rf_annuel) / dside if dside > 0 else np.nan
    def omega(x, seuil=0.0):
        g, l = (x - seuil).clip(lower=0).sum(), (seuil - x).clip(lower=0).sum()
        return g / l if l > 0 else np.nan
    def ulcer(x):
        return float(np.sqrt((dd(x) ** 2).mean()))
    reg = regression(p - rf, b - rf)
    beta = reg["beta"] if reg else np.nan
    alpha_j = reg["alpha"] * 252 if reg else np.nan
    bas = b < 0
    beta_bas = np.cov(p[bas], b[bas])[0, 1] / b[bas].var() if bas.sum() > 10 and b[bas].var() > 0 else np.nan
    hau = b > 0
    beta_haut = np.cov(p[hau], b[hau])[0, 1] / b[hau].var() if hau.sum() > 10 and b[hau].var() > 0 else np.nan
    te = (p - b).std(ddof=1) * np.sqrt(252)
    ir = (an(p) - an(b)) / te if te > 0 else np.nan
    sh_p, sh_b = sharpe_lo(p, rf_annuel), sharpe_lo(b, rf_annuel)
    m2 = rf_annuel + sh_p["sr_simple"] * vol(b) if pd.notna(sh_p["sr_simple"]) else np.nan
    haut, bas_ = b > 0, b < 0
    cap_h = p[haut].mean() / b[haut].mean() if haut.sum() and b[haut].mean() != 0 else np.nan
    cap_b = p[bas_].mean() / b[bas_].mean() if bas_.sum() and b[bas_].mean() != 0 else np.nan
    lignes = [
        ("Rendement annualisé", an(p), an(b), "Rendement composé ramené à un an."),
        ("Volatilité annualisée", vol(p), vol(b), "Écart-type des rendements quotidiens × √252."),
        ("Ratio de Sharpe (Lo)", sh_p["sr_lo"], sh_b["sr_lo"], "Excès sur le taux sans risque ÷ volatilité, annualisé avec la correction d'autocorrélation de Lo (2002)."),
        ("Ratio de Sortino", sortino(p), sortino(b), "Comme le Sharpe, mais seule la volatilité des baisses compte (Sortino et Price, 1994)."),
        ("Plus forte baisse", dd(p).min(), dd(b).min(), "Pire chute depuis un sommet sur la période."),
        ("Ratio de Calmar", (an(p) / -dd(p).min()) if dd(p).min() < 0 else np.nan, (an(b) / -dd(b).min()) if dd(b).min() < 0 else np.nan, "Rendement annualisé ÷ plus forte baisse."),
        ("Indice d'ulcère", ulcer(p), ulcer(b), "Profondeur et durée des baisses (Martin, 1987) : plus bas = moins pénible à détenir."),
        ("Ratio Omega", omega(p), omega(b), "Somme des gains ÷ somme des pertes quotidiennes (Keating et Shadwick, 2002) ; au-dessus de 1 = plus de gains que de pertes."),
        ("VaR 95 % historique (1 jour)", -np.percentile(p, 5), -np.percentile(b, 5), "Perte dépassée un jour sur vingt, lue dans l'historique."),
        ("VaR 95 % Cornish-Fisher (1 jour)", var_cornish_fisher(p), var_cornish_fisher(b), "VaR normale corrigée de l'asymétrie et des queues épaisses (Favre et Galeano, 2002)."),
        ("Expected Shortfall 95 % (1 jour)", -p[p <= np.percentile(p, 5)].mean(), -b[b <= np.percentile(b, 5)].mean(), "Perte moyenne les jours au-delà de la VaR."),
        ("Bêta", beta, 1.0, "Sensibilité aux mouvements de l'indice."),
        ("Bêta à la baisse", beta_bas, 1.0, "Bêta calculé seulement les jours de baisse de l'indice (Ang, Chen et Xing, 2006)."),
        ("Bêta à la hausse", beta_haut, 1.0, "Bêta calculé seulement les jours de hausse de l'indice. L'idéal : plus élevé que le bêta à la baisse. Chacun repose sur la moitié des séances, donc un petit écart entre les deux n'est pas significatif."),
        ("Alpha de Jensen (annualisé)", alpha_j, 0.0, "Rendement non expliqué par le bêta (Jensen, 1968)."),
        ("Ratio de Treynor", (an(p) - rf_annuel) / beta if beta and pd.notna(beta) and beta != 0 else np.nan, an(b) - rf_annuel, "Excès de rendement ÷ bêta : la rémunération du risque de marché."),
        ("Écart de suivi", te, 0.0, "Volatilité de l'écart quotidien avec l'indice."),
        ("Ratio d'information", ir, np.nan, "Surperformance annualisée ÷ écart de suivi : la régularité de la surperformance (Grinold et Kahn, 2000)."),
        ("M² (Modigliani)", m2, an(b), "Ton rendement s'il avait le risque de l'indice : la mesure la plus lisible (Modigliani et Modigliani, 1997)."),
        ("Capture haussière", cap_h, 1.0, "Part des hausses de l'indice que tu captes."),
        ("Capture baissière", cap_b, 1.0, "Part des baisses de l'indice que tu subis : sous 1, tu amortis."),
        ("Jours au-dessus de l'indice", float((p > b).mean()), np.nan, "Part des séances où tu fais mieux que l'indice."),
        ("R² ajusté face à l'indice", reg["r2_adj"] if reg else np.nan, 1.0, "Part de tes variations expliquée par l'indice, corrigée du nombre de variables."),
    ]
    return pd.DataFrame(lignes, columns=["ratio", "portefeuille", "indice", "definition"]), n


def hhi(poids):
    """Indice de Herfindahl-Hirschman sur les poids des lignes ramenés à 100 % du capital investi, et son inverse
    (nombre effectif de lignes)."""
    w = pd.Series(poids).dropna()
    w = w[w > 0]
    if w.empty:
        return np.nan, np.nan
    w = w / w.sum()
    h = float((w ** 2).sum())
    return h, 1 / h


# ---------------------------------------------------------------- qualité, investissement, résultats (v7)
def rentabilite_brute(marge_brute, actif_total):
    """Rentabilité brute de Novy-Marx (2013) : marge brute ÷ actif total. Plus prédictive des rendements futurs que le ROE,
    qui peut être gonflé par la dette. Non pertinente pour les banques et assureurs (pas de marge brute)."""
    try:
        gp, ta = float(marge_brute), float(actif_total)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(gp) or not np.isfinite(ta) or ta <= 0:
        return np.nan
    return gp / ta


def variation_annuelle(actuel, precedent):
    """Croissance d'une année sur l'autre (actif total : Cooper, Gulen et Schill, 2008 ; nombre d'actions : Pontiff et Woodgate, 2008)."""
    try:
        a, p = float(actuel), float(precedent)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(a) or not np.isfinite(p) or p <= 0:
        return np.nan
    return a / p - 1


def revision_prevision(actuelle, ancienne):
    """Révision du consensus de bénéfice par action, en % de la valeur absolue de l'ancienne prévision
    (Chan, Jegadeesh et Lakonishok, 1996)."""
    try:
        a, o = float(actuelle), float(ancienne)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(a) or not np.isfinite(o) or abs(o) < 1e-9:
        return np.nan
    return (a - o) / abs(o)


def signal_pead(surprise, date_publication, aujourd_hui, fenetre=60, seuil=0.02):
    """Dérive post-annonce (Bernard et Thomas, 1989) : le cours tend à continuer dans le sens de la surprise pendant
    environ 60 jours de bourse. Renvoie +1, −1 ou 0, et le nombre de jours calendaires depuis la publication."""
    if surprise is None or pd.isna(surprise) or date_publication is None or pd.isna(date_publication):
        return 0, np.nan
    jours = (pd.Timestamp(aujourd_hui) - pd.Timestamp(date_publication)).days
    if jours < 0 or jours > fenetre * 1.45:          # 60 jours de bourse ≈ 87 jours calendaires
        return 0, jours
    if surprise > seuil:
        return 1, jours
    if surprise < -seuil:
        return -1, jours
    return 0, jours


def active_share(poids_portefeuille, poids_indice):
    """Active share de Cremers et Petajisto (2009) : ½ Σ |w_portefeuille − w_indice|, les deux portefeuilles ramenés à 100 %
    de leur partie investie. 0 = copie de l'indice, 1 = aucun titre en commun."""
    wp = pd.Series(poids_portefeuille, dtype=float).dropna()
    wb = pd.Series(poids_indice, dtype=float).dropna()
    wp, wb = wp[wp > 0], wb[wb > 0]
    if wp.empty or wb.empty:
        return np.nan
    wp, wb = wp / wp.sum(), wb / wb.sum()
    tous = wp.index.union(wb.index)
    return float(0.5 * (wp.reindex(tous, fill_value=0) - wb.reindex(tous, fill_value=0)).abs().sum())


# ---------------------------------------------------------------- portefeuille modèle (v8)
def _z(x):
    """Score centré réduit, borné à ±3 pour qu'une valeur extrême ne décide pas seule."""
    x = pd.Series(x, dtype=float)
    e = x.std(ddof=0)
    return ((x - x.mean()) / e).clip(-3, 3) if e and e > 0 else x * 0.0


def nettoyer_prix(prix, seuil=0.5, limite=5):
    """Rend un tableau de cours utilisable ligne à ligne : retire les jours où moins de la moitié des titres cotent
    (jour férié général, ligne créée par les devises la nuit ou le week-end), puis reporte le dernier cours connu
    pendant au plus 5 séances (jour férié local). Le report n'utilise que le passé : aucune information du futur."""
    if prix is None or len(prix) == 0:
        return prix
    return prix.loc[prix.notna().mean(axis=1) >= seuil].ffill(limit=limite)


def signaux_modele(prix, date):
    """Signaux calculés uniquement avec les cours connus à la date donnée (aucune information du futur) :
    momentum 12-1 (Jegadeesh et Titman, 1993), faible volatilité sur un an (Blitz et van Vliet, 2007),
    proximité du plus haut sur 52 semaines (George et Hwang, 2004), asymétrie de bêta (bêta à la hausse moins bêta à la baisse,
    dans l'esprit de Ang, Chen et Xing, 2006 ; signal bruité, proposé pour être testé, pas recommandé par défaut)."""
    h = nettoyer_prix(prix.loc[:date])
    if len(h) < 275:
        return pd.DataFrame()
    f = h.iloc[-275:]
    ok = (f.notna().sum() >= 255) & f.iloc[-1].notna() & f.iloc[-22].notna() & f.iloc[-253].notna()
    f = f.loc[:, ok]
    if f.shape[1] == 0:
        return pd.DataFrame()
    r = f.pct_change(fill_method=None).iloc[-252:]
    # asymétrie de bêta : bêta les jours de hausse moins bêta les jours de baisse, face à la moyenne de l'univers
    X = r.fillna(0.0).values
    m = X.mean(axis=1)
    def _beta(masque):
        if masque.sum() < 30:
            return np.full(X.shape[1], np.nan)
        mc = m[masque] - m[masque].mean()
        den = mc @ mc
        return (X[masque] - X[masque].mean(axis=0)).T @ mc / den if den > 0 else np.full(X.shape[1], np.nan)
    # alpha : rendement non expliqué par le marché sur 12 mois moins le dernier, rapporté à son incertitude (statistique t).
    # C'est le « momentum résiduel » de Blitz, Huij et Martens (2011), ici avec un seul facteur (la moyenne de l'univers).
    Xw, mw = X[:-21], m[:-21]
    mc = mw - mw.mean()
    den = mc @ mc
    if den > 0 and len(mw) > 60:
        b = (Xw - Xw.mean(axis=0)).T @ mc / den
        a = Xw.mean(axis=0) - b * mw.mean()
        e = Xw - a - np.outer(mw, b)
        se = e.std(axis=0, ddof=2) / np.sqrt(len(mw))
        alpha_t = np.where(se > 0, a / np.where(se > 0, se, 1.0), np.nan)
    else:
        alpha_t = np.full(X.shape[1], np.nan)
    return pd.DataFrame({"alpha": pd.Series(alpha_t, index=f.columns),
                         "momentum": f.iloc[-22] / f.iloc[-253] - 1,
                         "vol": r.std(ddof=1) * np.sqrt(252),
                         "haut52": f.iloc[-1] / f.iloc[-252:].max(),
                         "asym_beta": pd.Series(_beta(m > 0) - _beta(m < 0), index=f.columns)})


def selection_modele(sig, secteurs, places, n=26, poids_signaux=None, cap_secteur=0.20, cap_place=0.30, forces=None):
    """Classe les titres par score composite (moyenne pondérée des scores centrés réduits) puis retient les n premiers
    en respectant un nombre maximal de titres par secteur et par place, pour que les plafonds restent atteignables."""
    ps = {"momentum": 1.0, "faible_vol": 1.0, "haut52": 0.0, "asym_beta": 0.0, "alpha": 0.0, **(poids_signaux or {})}
    if sig is None or sig.empty:
        return [], pd.Series(dtype=float)
    z = pd.DataFrame({"momentum": _z(sig["momentum"]), "faible_vol": -_z(sig["vol"]), "haut52": _z(sig["haut52"]),
                      "asym_beta": _z(sig["asym_beta"]).fillna(0.0) if "asym_beta" in sig else 0.0,
                      "alpha": _z(sig["alpha"]).fillna(0.0) if "alpha" in sig else 0.0})
    tot = sum(abs(v) for v in ps.values())
    score = sum(ps[k] * z[k] for k in z.columns) / tot if tot > 0 else z.mean(axis=1) * 0.0
    score = score.dropna().sort_values(ascending=False, kind="mergesort")
    n = int(min(n, len(score)))
    max_s, max_p = max(1, int(np.floor(cap_secteur * n + 1e-9))), max(1, int(np.floor(cap_place * n + 1e-9)))
    cs, cp, choisis = {}, {}, []
    # titres imposés (par exemple l'or, retenu pour son rôle d'amortisseur et non pour son score) : ils prennent une place, le classement remplit le reste
    imposes = [t for t in (forces or []) if t in score.index]
    for t in imposes + [t for t in score.index if t not in imposes]:
        s_, p_ = secteurs.get(t, "?"), places.get(t, "?")
        if cs.get(s_, 0) >= max_s or cp.get(p_, 0) >= max_p:
            continue
        choisis.append(t)
        cs[s_], cp[p_] = cs.get(s_, 0) + 1, cp.get(p_, 0) + 1
        if len(choisis) == n:
            break
    return choisis, score


def poids_modele(prix, date, titres, secteurs, places, mode="mdp", borne=0.10, cap_secteur=0.20, cap_place=0.30, retrecissement=0.2):
    """Poids fondés sur le risque seul, estimé sur un an de cours connus à la date. La matrice de covariance est
    rétrécie vers sa diagonale (dans l'esprit de Ledoit et Wolf, 2004) pour limiter l'erreur d'estimation.
    Chaque ligne retenue reçoit au moins 1,5 % (ou la moitié du poids égal), pour ne pas garder de ligne vide."""
    n = len(titres)
    if n == 0:
        return pd.Series(dtype=float)
    if mode == "egal" or n < 3:
        return pd.Series(1 / n, index=titres)
    r = prix.loc[:date, titres].pct_change(fill_method=None).iloc[-252:]
    S = r.cov().values * 252
    if not np.isfinite(S).all():
        return pd.Series(1 / n, index=titres)
    S = (1 - retrecissement) * S + retrecissement * np.diag(np.diag(S))
    sec, pla = np.array([secteurs.get(t, "?") for t in titres]), np.array([places.get(t, "?") for t in titres])
    groupes = [(sec == g, cap_secteur) for g in np.unique(sec)] + [(pla == g, cap_place) for g in np.unique(pla)]
    try:
        w = optimiser(S, mode, borne, groupes, plancher=min(0.015, 0.5 / n), maxiter=150, ftol=1e-9)    # aucune ligne « fantôme » à 0 %
    except Exception:
        w = np.full(n, 1 / n)
    if not np.isfinite(w).all():
        w = np.full(n, 1 / n)
    return pd.Series(w, index=titres)


def _dates_rebal(prix):
    """Dernier jour de bourse de chaque mois, plus la dernière date disponible."""
    fins = prix.groupby([prix.index.year, prix.index.month]).tail(1).index
    return list(fins) if fins[-1] == prix.index[-1] else list(fins) + [prix.index[-1]]


def backtest_modele(prix, secteurs, places, n=26, poids_signaux=None, mode="mdp", borne=0.10, cap_secteur=0.20, cap_place=0.30, forces=None):
    """Test hors échantillon « pas à pas » : à chaque fin de mois, le portefeuille est reconstruit avec les seules données
    connues ce jour-là, puis détenu le mois suivant (poids fixes dans le mois). Renvoie les rendements quotidiens,
    les compositions successives, la rotation moyenne et le rendement de l'univers à poids égaux (mêmes titres éligibles)."""
    prix = nettoyer_prix(prix)
    R = prix.pct_change(fill_method=None)
    dates = _dates_rebal(prix)
    morceaux, ew, compos, rot, w_prec = [], [], {}, [], None
    for d0, d1 in zip(dates[:-1], dates[1:]):
        sig = signaux_modele(prix, d0)
        if sig.empty or len(sig) < max(6, min(n, 10)):
            continue
        titres, _ = selection_modele(sig, secteurs, places, n, poids_signaux, cap_secteur, cap_place, forces)
        if len(titres) < 3:
            continue
        w = poids_modele(prix, d0, titres, secteurs, places, mode, borne, cap_secteur, cap_place)
        fen = R.loc[(R.index > d0) & (R.index <= d1)]
        if fen.empty:
            continue
        morceaux.append(fen[titres].fillna(0.0) @ w)
        ew.append(fen[sig.index].mean(axis=1, skipna=True))
        compos[d0] = w
        if w_prec is not None:
            tous = w.index.union(w_prec.index)
            rot.append(0.5 * (w.reindex(tous, fill_value=0) - w_prec.reindex(tous, fill_value=0)).abs().sum())
        w_prec = w
    if not morceaux:
        return None
    return {"r": pd.concat(morceaux), "r_egal": pd.concat(ew).fillna(0.0), "compositions": compos,
            "rotation": float(np.mean(rot)) if rot else np.nan}


def singes_modele(prix, secteurs, places, dates, n=26, cap_secteur=0.20, cap_place=0.30, n_singes=1000, graine=7):
    """Le monde du hasard pour le portefeuille modèle : n_singes portefeuilles qui, aux mêmes dates, tirent au sort
    le même nombre de titres dans le même univers éligible, avec les mêmes limites par secteur et par place, à poids égaux.
    Renvoie la matrice des rendements quotidiens (jours × singes)."""
    rng = np.random.default_rng(graine)
    prix = nettoyer_prix(prix)
    R = prix.pct_change(fill_method=None)
    dates = sorted(dates) + [prix.index[-1]]
    blocs = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        if d1 <= d0:
            continue
        sig = signaux_modele(prix, d0)
        fen = R.loc[(R.index > d0) & (R.index <= d1)]
        if sig.empty or fen.empty:
            continue
        elig = list(sig.index)
        k = int(min(n, len(elig)))
        max_s, max_p = max(1, int(np.floor(cap_secteur * k + 1e-9))), max(1, int(np.floor(cap_place * k + 1e-9)))
        sec = [secteurs.get(t, "?") for t in elig]
        pla = [places.get(t, "?") for t in elig]
        S = np.zeros((n_singes, len(elig)))
        for i in range(n_singes):
            cs, cp, pris = {}, {}, 0
            for j in rng.permutation(len(elig)):
                if cs.get(sec[j], 0) >= max_s or cp.get(pla[j], 0) >= max_p:
                    continue
                S[i, j] = 1.0
                cs[sec[j]], cp[pla[j]] = cs.get(sec[j], 0) + 1, cp.get(pla[j], 0) + 1
                pris += 1
                if pris == k:
                    break
            if pris:
                S[i] /= pris
        blocs.append(pd.DataFrame(fen[elig].fillna(0.0).values @ S.T, index=fen.index))
    return pd.concat(blocs) if blocs else pd.DataFrame()


def bilan_serie(r):
    """Rendement annualisé, volatilité, Sharpe simple et plus forte baisse d'une série de rendements quotidiens."""
    x = pd.Series(r).dropna()
    if len(x) < 20:
        return {"rend": np.nan, "vol": np.nan, "sharpe": np.nan, "baisse": np.nan, "total": np.nan}
    v = (1 + x).cumprod()
    an = v.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std(ddof=1) * np.sqrt(252)
    return {"rend": float(an), "vol": float(vol), "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else np.nan,
            "baisse": float((v / v.cummax() - 1).min()), "total": float(v.iloc[-1] - 1)}


def confirmations_modele(r, r_egal, r_bench, singes, n_essais=10, n_boot=1000):
    """Batterie de tests sur un portefeuille modèle. Chaque test répond à une question différente ; aucun ne suffit seul."""
    out = {}
    x = pd.Series(r).dropna()
    # 1 et 2 : face au hasard (mêmes dates, même univers, mêmes règles)
    if singes is not None and not singes.empty:
        sg = singes.reindex(x.index).fillna(0.0)
        tot = (1 + sg).prod() - 1
        sh = sg.mean() / sg.std(ddof=1) * np.sqrt(252)
        b = bilan_serie(x)
        out["singes_total"], out["singes_sharpe"] = tot, sh
        out["p_rendement"] = float((tot >= b["total"]).mean())
        out["p_sharpe"] = float((sh >= b["sharpe"]).mean())
    # 3 : alpha face à l'indice, bootstrap par blocs
    if r_bench is not None:
        bo = bootstrap_alpha(x, pd.Series(r_bench), n_boot=n_boot, longueur_bloc=5)
        if bo:
            out["alpha_an"], out["p_alpha"], out["beta"] = bo["alpha"] * 252, bo["p_boot"], bo["beta"]
    # 4 : Sharpe dégonflé du nombre d'essais
    ds = sharpe_degonfle(x, n_essais)
    if ds:
        out["dsr"], out["sr_seuil"] = ds["dsr"], ds["sr_seuil_annuel"]
    # 5 : stabilité dans le temps, face à l'univers à poids égaux
    z = pd.concat([x.rename("m"), pd.Series(r_egal).rename("e")], axis=1).dropna()
    if len(z) > 60:
        an = z.groupby(z.index.year).agg(lambda c: (1 + c).prod() - 1)
        nb = z.groupby(z.index.year)["m"].count()
        an = an[nb >= 60]
        out["annees"] = an
        out["part_annees"] = float((an["m"] > an["e"]).mean()) if len(an) else np.nan
        moitie = len(z) // 2
        out["moities"] = [float((1 + z["m"].iloc[:moitie]).prod() - (1 + z["e"].iloc[:moitie]).prod()),
                          float((1 + z["m"].iloc[moitie:]).prod() - (1 + z["e"].iloc[moitie:]).prod())]
    return out


def stop_suiveur(prix, date_achat, vol_annuelle, k=2.5, plancher=0.08, plafond=0.30):
    """Stop suiveur proportionnel à la volatilité du titre : la distance vaut k fois sa volatilité mensuelle
    (bornée entre 8 % et 30 %), mesurée depuis le plus haut atteint depuis l'achat. Un stop fixe à 5 % se déclenche
    sans arrêt sur un titre volatil ; un stop n'a d'intérêt que si les cours ont une tendance (Kaminski et Lo, 2014).
    Renvoie le recul depuis le plus haut, le seuil, la date du plus haut et si le stop est touché."""
    x = pd.Series(prix).dropna()
    if date_achat is not None and pd.notna(date_achat):
        x = x.loc[x.index >= pd.Timestamp(date_achat)]
    if len(x) < 2 or vol_annuelle is None or pd.isna(vol_annuelle):
        return None
    seuil = float(np.clip(k * vol_annuelle / np.sqrt(12), plancher, plafond))
    haut = x.max()
    recul = float(x.iloc[-1] / haut - 1)
    return {"recul": recul, "seuil": -seuil, "date_haut": x.idxmax(), "touche": recul <= -seuil}


# ---------------------------------------------------------------- scénarios politiques (v9)
def sensibilite_facteur(R, r_marche, facteur, fenetre=504):
    """Pour chaque titre : r = a + b_marché × marché + b_facteur × facteur. Le facteur est un rendement relatif
    (par exemple CAC 40 moins Europe) : b_facteur mesure ce que le titre gagne ou perd quand la France fait
    1 point de mieux ou de moins bien que le reste de l'Europe, à marché européen inchangé."""
    X = pd.DataFrame({"marche": r_marche, "facteur": facteur}).dropna().iloc[-fenetre:]
    lignes = {}
    for t in R.columns:
        y = R[t].reindex(X.index)
        ok = y.notna()
        if ok.sum() < 120:
            continue
        reg = regression_multi(y[ok], X[ok])
        if reg is None:
            continue
        lignes[t] = {"beta_marche": float(reg["betas"]["marche"]), "beta_facteur": float(reg["betas"]["facteur"]),
                     "t_facteur": float(reg["t"]["facteur"]) if "t" in reg and "facteur" in reg["t"] else np.nan,
                     "r2_adj": float(reg["r2_adj"]), "n": int(ok.sum())}
    return pd.DataFrame(lignes).T


def prix_binaire(p, choc, beta=1.0):
    """Événement à deux issues déjà en partie dans les cours. Si le marché donne une probabilité p au scénario de tension
    et que l'écart total de valeur entre les deux issues vaut « choc », alors le cours bouge de :
    (1 − p) × choc si le scénario se réalise, − p × choc s'il est écarté, et de 0,10 × choc quand p monte de 10 points."""
    p = float(np.clip(p, 0.0, 1.0))
    return {"realise": (1 - p) * choc * beta, "ecarte": -p * choc * beta, "dix_points": 0.10 * choc * beta}


def reaction_aux_cotes(facteur, proba):
    """Le facteur réagit-il aux variations de la probabilité ? Régression du rendement quotidien du facteur sur la variation
    quotidienne de la probabilité. La pente est l'écart total implicite entre les deux issues (le « choc »)."""
    z = pd.DataFrame({"f": facteur, "dp": pd.Series(proba).diff()}).dropna()
    z = z[z["dp"].abs() > 0]
    if len(z) < 30 or z["dp"].var() == 0:
        return None
    reg = regression(z["f"], z["dp"])
    if reg is None:
        return None
    return {"pente": float(reg["beta"]), "t": float(reg.get("t_beta", np.nan)), "r2": float(reg["r2"]), "n": int(len(z))}


# ================================================================== v11 : scénarios
def _blocs(n_obs, longueur, bloc, rng, n_tirages=None):
    """Indices d'un bootstrap par blocs circulaire : des morceaux de `bloc` jours consécutifs tirés au hasard et mis bout à bout.
    Garder des blocs conserve l'enchaînement des jours (les séries de hausses ou de baisses), ce qu'un tirage jour par jour détruirait."""
    k = int(np.ceil(longueur / bloc))
    forme = (k,) if n_tirages is None else (n_tirages, k)
    dep = rng.integers(0, n_obs, forme)
    idx = (dep[..., None] + np.arange(bloc)) % n_obs
    return idx.reshape(*forme[:-1], -1)[..., :longueur]


def robustesse_modele(prix, date, secteurs, places, n=26, poids_signaux=None, mode="mdp", borne=0.10, forces=None,
                      n_scen=200, bloc=21, graine=11, cap_secteur=0.20, cap_place=0.30):
    """La sélection tient-elle si l'année écoulée s'était déroulée un peu autrement ? (rééchantillonnage, dans l'esprit de Michaud, 1998)
    On remélange n_scen fois les mois de l'année écoulée (mêmes jours pour tous les titres, donc mêmes liens entre eux), et on refait
    à chaque fois la sélection et les poids. Renvoie la fréquence de sélection de chaque titre, les poids moyens des titres retenus
    sur les vraies données, leur dispersion, et la stabilité (fréquence moyenne des titres retenus)."""
    h = nettoyer_prix(prix.loc[:date])
    sig0 = signaux_modele(h, h.index[-1])
    if sig0.empty:
        return None
    base, _ = selection_modele(sig0, secteurs, places, n, poids_signaux, cap_secteur, cap_place, forces)
    if len(base) < 3:
        return None
    f = h.iloc[-275:][list(sig0.index)]
    L = np.log1p(f.pct_change(fill_method=None).iloc[1:].fillna(0.0).values)
    T = len(L)
    rng = np.random.default_rng(graine)
    freq = pd.Series(0.0, index=f.columns)
    W = []
    for _ in range(n_scen):
        idx = _blocs(T, T, bloc, rng)
        P = pd.DataFrame(np.exp(np.vstack([np.zeros(L.shape[1]), np.cumsum(L[idx], axis=0)])), index=f.index, columns=f.columns)
        sg = signaux_modele(P, P.index[-1])
        if sg.empty:
            continue
        tit, _ = selection_modele(sg, secteurs, places, n, poids_signaux, cap_secteur, cap_place, forces)
        freq[tit] += 1
        W.append(poids_modele(P, P.index[-1], base, secteurs, places, mode, borne, cap_secteur, cap_place).values)
    if not W:
        return None
    W = np.array(W)
    return {"frequence": freq / len(W), "base": base, "poids_moyens": pd.Series(W.mean(axis=0), index=base),
            "poids_ecart": pd.Series(W.std(axis=0), index=base), "stabilite": float((freq / len(W)).reindex(base).mean()), "n": len(W)}


def simulation_horizon(rendements, jours, n_sim=5000, bloc=10, graine=5, centrer_sur=None):
    """Ce que les `jours` prochaines séances pourraient donner, par bootstrap par blocs des rendements passés.
    `rendements` : une colonne par portefeuille, mêmes dates ; les mêmes jours sont tirés pour toutes les colonnes, donc les liens
    entre portefeuilles sont conservés. `centrer_sur` : nom d'une colonne dont le rendement moyen est imposé à toutes les autres
    (hypothèse « aucun avantage » : seuls le risque et la diversification distinguent alors les portefeuilles).
    Renvoie deux tableaux (simulations × portefeuilles) : le rendement total et la plus forte baisse en cours de route."""
    X = pd.DataFrame(rendements).dropna()
    if len(X) < 120 or jours < 5:
        return None
    V = X.values.copy()
    if centrer_sur is not None and centrer_sur in X.columns:
        V = V - V.mean(axis=0) + V[:, list(X.columns).index(centrer_sur)].mean()
    L = np.log1p(np.clip(V, -0.95, None))
    rng = np.random.default_rng(graine)
    tot, creux = [], []
    for debut in range(0, n_sim, 1000):                      # par paquets, pour ne pas occuper trop de mémoire
        m = min(1000, n_sim - debut)
        chemins = np.cumsum(L[_blocs(len(L), jours, bloc, rng, m)], axis=1)          # (m, jours, k)
        tot.append(np.expm1(chemins[:, -1, :]))
        sommet = np.maximum.accumulate(np.maximum(chemins, 0.0), axis=1)
        creux.append(np.expm1((chemins - sommet).min(axis=1)))
    return pd.DataFrame(np.vstack(tot), columns=X.columns), pd.DataFrame(np.vstack(creux), columns=X.columns)
