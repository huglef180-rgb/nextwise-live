"""
NextWise — suivi du portefeuille en direct (Streamlit)

Données :
- Journal de transactions et fiches des lignes : Google Sheet (onglets « transactions » et « lignes »)
  si SHEET_ID est défini dans les secrets Streamlit, sinon les fichiers transactions.csv et lignes.csv.
- Cours : Yahoo Finance via yfinance (différés d'environ 15 minutes), convertis en euros.
- Les chiffres officiels du concours restent ceux de NextWise.
"""
print("[nextwise] le script démarre (avant les imports)", flush=True)
import datetime as dt
import os
import urllib.parse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

print("[nextwise] imports terminés", flush=True)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

st.set_page_config(page_title="NextWise · suivi en direct", page_icon="📈", layout="wide")


def log(msg):
    """Écrit dans les journaux Streamlit (Manage app) pour suivre le démarrage."""
    print(f"[nextwise {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


log("démarrage du script")

# ---------------------------------------------------------------- paramètres
TZ = ZoneInfo("Europe/Brussels")
LIM = {
    "ucits_max": 0.10,      # aucune ligne au-dessus de 10 %
    "ucits_seuil": 0.05,    # les lignes au-dessus de 5 %…
    "ucits_somme": 0.40,    # …ne dépassent pas 40 % ensemble
    "secteur": 0.20,        # un secteur ≤ 20 % du capital
    "place": 0.30,          # une place de cotation ≤ 30 %
    "alerte_poids": 0.09,   # alerte avant la limite UCITS
    "alerte_somme": 0.35,
    "alerte_jour": -0.05,   # baisse du jour qui déclenche une alerte
    "momentum": -0.20,      # momentum 12-1 sous lequel la ligne sort
    "vol_reduite": 0.40,    # vol EWMA au-dessus de laquelle le poids est réduit
}
BENCH = {"^N100": "Euronext 100", "^STOXX": "Stoxx Europe 600"}
BENCH_PRINCIPAL = "^N100"
DEVISE_SUFFIXE = {".OL": "NOK", ".ST": "SEK", ".CO": "DKK", ".SW": "CHF", ".L": "GBp", ".WA": "PLN"}
CRISES = {
    "Krach du Covid (2020)": ("2020-02-19", "2020-03-23"),
    "Invasion de l'Ukraine (2022)": ("2022-02-10", "2022-03-07"),
    "Inflation et hausse des taux (2022)": ("2022-01-04", "2022-09-29"),
    "Crise bancaire SVB / Credit Suisse (2023)": ("2023-03-08", "2023-03-20"),
    "DeepSeek, choc sur l'IA (2025)": ("2025-01-24", "2025-01-28"),
    "Droits de douane américains (2025)": ("2025-04-02", "2025-04-08"),
}
MOCK = os.environ.get("NEXTWISE_MOCK") == "1"
ICI = os.path.dirname(os.path.abspath(__file__))

C_BON, C_MAUVAIS, C_ATTENTION, C_NEUTRE = "#1a9850", "#d73027", "#e69500", "#8c8c8c"


# ---------------------------------------------------------------- utilitaires
def num(x):
    """Lit un nombre écrit à la française ou à l'anglaise ; vide → NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\xa0", "").replace(" ", "").replace("€", "")
    if s == "":
        return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def eur(x, d=0):
    if x is None or pd.isna(x):
        return "—"
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return f"{s} €"


def pct(x, d=1, signe=False):
    if x is None or pd.isna(x):
        return "—"
    s = f"{x * 100:+.{d}f}" if signe else f"{x * 100:.{d}f}"
    return s.replace(".", ",") + " %"


def devise_de(ticker, override=None):
    if isinstance(override, str) and override.strip():
        return override.strip()
    for suf, dev in DEVISE_SUFFIXE.items():
        if ticker.upper().endswith(suf):
            return dev
    return "EUR"


def secrets_get(cle):
    try:
        return st.secrets.get(cle)
    except Exception:
        return None


# ---------------------------------------------------------------- tables de l'utilisateur
@st.cache_data(ttl=60, show_spinner=False)
def charger_tables():
    sheet = secrets_get("SHEET_ID")
    if sheet:
        base = f"https://docs.google.com/spreadsheets/d/{sheet}/gviz/tq?tqx=out:csv&sheet="
        tx = pd.read_csv(base + urllib.parse.quote("transactions"), dtype=str)
        li = pd.read_csv(base + urllib.parse.quote("lignes"), dtype=str)
        source = "Google Sheet"
    else:
        tx = pd.read_csv(os.path.join(ICI, "transactions.csv"), dtype=str)
        li = pd.read_csv(os.path.join(ICI, "lignes.csv"), dtype=str)
        source = "fichiers du dépôt"
    tx.columns = [c.strip().lower() for c in tx.columns]
    li.columns = [c.strip().lower() for c in li.columns]
    tx = tx.dropna(how="all")
    li = li.dropna(how="all")
    for c in ["date", "ticker", "type", "quantite", "prix", "frais", "commentaire"]:
        if c not in tx.columns:
            tx[c] = None
    for c in ["ticker", "nom", "poche", "secteur", "place", "poids_cible", "these", "menace", "sortie", "evenement", "devise"]:
        if c not in li.columns:
            li[c] = None
    tx["ticker"] = tx["ticker"].fillna("").str.strip().str.upper()
    tx["type"] = tx["type"].fillna("").str.strip().str.lower()
    brut = tx["date"].fillna("").astype(str).str.strip().str[:10]
    iso = pd.to_datetime(brut, format="%Y-%m-%d", errors="coerce")       # 2026-11-03
    fr_ = pd.to_datetime(brut, format="%d/%m/%Y", errors="coerce")       # 03/11/2026 (Google Sheet en français)
    tx["date"] = iso.fillna(fr_)
    for c in ["quantite", "prix", "frais"]:
        tx[c] = tx[c].map(num)
    tx["frais"] = tx["frais"].fillna(0.0)
    tx = tx.sort_values("date", kind="stable").reset_index(drop=True)
    li["ticker"] = li["ticker"].fillna("").str.strip().str.upper()
    li = li[li["ticker"] != ""].drop_duplicates("ticker", keep="last").set_index("ticker")
    li["poids_cible"] = li["poids_cible"].map(num)
    return tx, li, source


def traiter_journal(tx):
    """Rejoue le journal : positions, prix de revient moyen, cash, ventes réalisées."""
    pos = {}  # ticker -> [quantité, coût total]
    cash, apports = 0.0, 0.0
    ventes, erreurs, instantanes = [], [], []
    for i, r in tx.iterrows():
        t, typ, q, p, f = r["ticker"], r["type"], r["quantite"], r["prix"], r["frais"]
        if pd.isna(r["date"]):
            erreurs.append(f"Ligne {i + 2} du journal : date illisible, ligne ignorée.")
            continue
        if typ == "apport":
            cash += p if not pd.isna(p) else 0.0
            apports += p if not pd.isna(p) else 0.0
        elif typ == "retrait":
            cash -= p if not pd.isna(p) else 0.0
            apports -= p if not pd.isna(p) else 0.0
        elif typ == "achat":
            if not t or pd.isna(q) or pd.isna(p):
                erreurs.append(f"Ligne {i + 2} : achat incomplet (ticker, quantité ou prix manquant).")
                continue
            q0, c0 = pos.get(t, [0.0, 0.0])
            pos[t] = [q0 + q, c0 + q * p + f]
            cash -= q * p + f
        elif typ == "vente":
            if not t or pd.isna(q) or pd.isna(p):
                erreurs.append(f"Ligne {i + 2} : vente incomplète.")
                continue
            q0, c0 = pos.get(t, [0.0, 0.0])
            if q > q0 + 1e-9:
                erreurs.append(f"Ligne {i + 2} : vente de {q:g} {t} alors que tu n'en as que {q0:g}.")
                q = q0
            if q <= 0:
                continue
            pru = c0 / q0
            instantanes.append((r["date"], t, {k: (v[0], v[1] / v[0]) for k, v in pos.items() if v[0] > 0 and k != t}))
            ventes.append({"date": r["date"], "ticker": t, "quantite": q, "prix": p, "pru": pru,
                           "resultat": q * (p - pru) - f, "resultat_pct": p / pru - 1})
            pos[t] = [q0 - q, c0 - pru * q]
            cash += q * p - f
        elif typ == "frais":
            cash -= f if f else (p if not pd.isna(p) else 0.0)
        elif typ == "dividende":
            montant = (q * p) if not pd.isna(q) and not pd.isna(p) else (p if not pd.isna(p) else 0.0)
            cash += montant - f
        else:
            erreurs.append(f"Ligne {i + 2} : type « {r['type']} » inconnu (Achat, Vente, Frais, Dividende, Apport, Retrait).")
    positions = pd.DataFrame(
        [{"ticker": k, "quantite": v[0], "pru": v[1] / v[0]} for k, v in pos.items() if v[0] > 1e-9]
    )
    if positions.empty:
        positions = pd.DataFrame(columns=["ticker", "quantite", "pru"])
    return positions.set_index("ticker"), cash, apports, pd.DataFrame(ventes), erreurs, instantanes


# ---------------------------------------------------------------- cours
def _faux_historique(tickers, jours=1900):
    """Cours simulés, uniquement pour tester l'appli hors ligne (NEXTWISE_MOCK=1).
    Chaque série est calée sur le prix d'achat du journal pour rester réaliste."""
    idx = pd.bdate_range(end=pd.Timestamp.now(tz=TZ).normalize().tz_localize(None), periods=jours)
    rng = np.random.default_rng(7)
    marche = rng.normal(0.0003, 0.011, len(idx))
    try:
        ancre = pd.read_csv(os.path.join(ICI, "transactions.csv")).dropna(subset=["ticker"]).groupby("ticker")["prix"].last()
    except Exception:
        ancre = pd.Series(dtype=float)
    out = {}
    for j, t in enumerate(tickers):
        if t.endswith("=X"):
            out[t] = 11.5 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
            continue
        b = 0.5 + (j % 7) * 0.15
        r = b * marche + rng.normal(0, 0.012 + (j % 5) * 0.003, len(idx))
        p = np.exp(np.cumsum(r))
        niveau = ancre.get(t, 1000.0 if t.startswith("^") else 50.0)
        if devise_de(t) == "NOK":
            niveau *= 11.5
        out[t] = p / p[-3] * niveau
    df = pd.DataFrame(out, index=idx)
    return df, df.copy()


@st.cache_data(ttl=3600, show_spinner="Téléchargement de l'historique des cours…")
def historique(tickers):
    tickers = sorted(set(tickers))
    if MOCK or yf is None:
        return _faux_historique(tickers)
    log(f"téléchargement historique : {len(tickers)} tickers")
    t0 = dt.datetime.now()
    try:
        data = yf.download(tickers, period="7y", interval="1d", auto_adjust=False, progress=False, threads=False, timeout=20)
    except Exception as e:
        log(f"échec du téléchargement historique : {type(e).__name__}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    log(f"historique reçu en {(dt.datetime.now() - t0).seconds} s, {0 if data is None else data.shape}")
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close, adj = data["Close"], data["Adj Close"]
    else:
        close, adj = data[["Close"]].rename(columns={"Close": tickers[0]}), data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    adj.index = pd.to_datetime(adj.index).tz_localize(None)
    return close.sort_index(), adj.sort_index()


@st.cache_data(ttl=60, show_spinner=False)
def intraday(tickers):
    tickers = sorted(set(tickers))
    if MOCK or yf is None:
        h, _ = _faux_historique(tickers)
        idx = pd.date_range(pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(hours=9), periods=60, freq="5min")
        rng = np.random.default_rng(int(dt.datetime.now().minute))
        df = pd.DataFrame({t: h[t].iloc[-2] * np.exp(np.cumsum(rng.normal(0, 0.0015, len(idx)))) for t in tickers}, index=idx)
        return df
    try:
        data = yf.download(tickers, period="1d", interval="5m", auto_adjust=False, progress=False, threads=False, timeout=20)
    except Exception as e:
        log(f"échec du téléchargement intrajournalier : {type(e).__name__}: {e}")
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": tickers[0]})
    idx = pd.to_datetime(close.index)
    close.index = idx.tz_convert(TZ) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(TZ)
    return close


def en_euros(df, devises):
    """Convertit chaque colonne en euros avec les paires EURxxx=X présentes dans df."""
    out = df.copy()
    for t, dev in devises.items():
        if t not in out.columns or dev == "EUR":
            continue
        if dev == "GBp":
            paire, facteur = "EURGBP=X", 100.0
        else:
            paire, facteur = f"EUR{dev}=X", 1.0
        if paire in df.columns:
            out[t] = df[t] / facteur / df[paire].ffill()
    return out


# ---------------------------------------------------------------- calculs de risque
def stats_lignes(adj_eur, bench):
    r = adj_eur.pct_change(fill_method=None)
    an = r.iloc[-252:]
    rb = r[bench].iloc[-252:] if bench in r else None
    lam = 0.94
    rows = {}
    for t in adj_eur.columns:
        s = adj_eur[t].dropna()
        x = an[t].dropna()
        if len(x) < 60:
            rows[t] = {}
            continue
        w = lam ** np.arange(len(x))[::-1]
        ewma = np.sqrt((w * (x - 0) ** 2).sum() / w.sum() * 252)
        beta = np.nan
        if rb is not None:
            z = pd.concat([x, rb], axis=1).dropna()
            if len(z) > 60:
                beta = 0.67 * np.cov(z.iloc[:, 0], z.iloc[:, 1])[0, 1] / z.iloc[:, 1].var() + 0.33
        mom = s.iloc[-22] / s.iloc[-253] - 1 if len(s) > 253 else np.nan
        haut52 = s.iloc[-1] / s.iloc[-252:].max() if len(s) > 20 else np.nan
        maxj = x.iloc[-21:].max()
        rows[t] = {"vol": x.std() * np.sqrt(252), "vol_ewma": ewma, "beta": beta, "momentum": mom, "haut52": haut52, "max_jour": maxj}
    return pd.DataFrame(rows).T


def risque_portefeuille(poids, adj_eur, bench):
    """poids : Series ticker -> poids sur la valeur totale (le cash a un poids implicite)."""
    r = adj_eur.pct_change(fill_method=None).iloc[-252:]
    t = [x for x in poids.index if x in r.columns and r[x].notna().sum() > 120]
    if not t:
        return None
    w = poids.reindex(t).fillna(0).values
    R = r[t].fillna(0)
    cov = R.cov().values * 252
    var = float(w @ cov @ w)
    vol = np.sqrt(var)
    sig = np.sqrt(np.diag(cov))
    dr = float((w * sig).sum() / vol) if vol > 0 else np.nan
    contrib = w * (cov @ w) / var if var > 0 else w * 0
    rp = R.values @ w
    var95 = -np.percentile(rp, 5)
    es95 = -rp[rp <= -var95].mean() if (rp <= -var95).any() else np.nan
    beta = np.nan
    if bench in r:
        z = pd.concat([pd.Series(rp, index=R.index), r[bench]], axis=1).dropna()
        if len(z) > 60:
            beta = np.cov(z.iloc[:, 0], z.iloc[:, 1])[0, 1] / z.iloc[:, 1].var()
    return {"vol": vol, "dr": dr, "var95": var95, "es95": es95, "beta": beta,
            "contrib": pd.Series(contrib, index=t), "couverture": float(poids.reindex(t).sum() / max(poids.sum(), 1e-9)),
            "corr": R.corr()}


def stress(poids, adj_eur, bench):
    rows = []
    for nom, (a, b) in CRISES.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        s = adj_eur.loc[(adj_eur.index >= a) & (adj_eur.index <= b)]
        if len(s) < 2:
            continue
        ret = s.ffill().iloc[-1] / s.bfill().iloc[0] - 1
        dispo = [t for t in poids.index if t in ret.index and pd.notna(ret[t])]
        if not dispo:
            continue
        w = poids.reindex(dispo)
        investi = poids.sum()
        # lignes sans historique : on leur prête le rendement moyen des autres
        r_ptf = float((w * ret[dispo]).sum() / w.sum() * investi)
        rows.append({"scenario": nom, "portefeuille": r_ptf, "reference": ret.get(bench, np.nan),
                     "couverture": float(w.sum() / investi) if investi else np.nan})
    return pd.DataFrame(rows)


def contraintes(df):
    """df : colonnes poids, secteur, place (une ligne par titre)."""
    out = []
    if df.empty:
        return pd.DataFrame(columns=["regle", "niveau", "limite", "etat"])
    m = df["poids"].idxmax()
    out.append(("UCITS : plus grosse ligne (" + str(df.loc[m, "nom"]) + ")", df["poids"].max(), LIM["ucits_max"], LIM["alerte_poids"]))
    s5 = df.loc[df["poids"] > LIM["ucits_seuil"], "poids"].sum()
    out.append(("UCITS : lignes au-dessus de 5 %, cumulées", s5, LIM["ucits_somme"], LIM["alerte_somme"]))
    sec = df.groupby(df["secteur"].fillna("Non renseigné"))["poids"].sum()
    out.append(("Premier secteur (" + sec.idxmax() + ")", sec.max(), LIM["secteur"], LIM["secteur"] - 0.02))
    pl = df.groupby(df["place"].fillna("Non renseignée"))["poids"].sum()
    out.append(("Première place de cotation (" + pl.idxmax() + ")", pl.max(), LIM["place"], LIM["place"] - 0.03))
    rows = []
    for regle, niveau, limite, alerte in out:
        etat = "Dépassée" if niveau > limite + 1e-9 else ("Proche" if niveau > alerte else "Respectée")
        rows.append({"regle": regle, "niveau": niveau, "limite": limite, "etat": etat})
    return pd.DataFrame(rows)


def couleur_etat(v):
    return {"Dépassée": f"color:{C_MAUVAIS};font-weight:600", "Proche": f"color:{C_ATTENTION};font-weight:600",
            "Respectée": f"color:{C_BON}"}.get(v, "")


def signe_couleur(v):
    if pd.isna(v):
        return ""
    return f"color:{C_BON}" if v > 0 else (f"color:{C_MAUVAIS}" if v < 0 else "")



# ---------------------------------------------------------------- chance ou talent
POIDS_BASE, POIDS_REDUIT = 0.0414, 0.025


@st.cache_data(ttl=3600, show_spinner=False)
def charger_univers():
    u = pd.read_csv(os.path.join(ICI, "univers.csv"), dtype=str)
    u["ticker"] = u["ticker"].str.strip().str.upper()
    return u.drop_duplicates("ticker").set_index("ticker")


def prix_au(df, date):
    """Dernier cours connu à cette date (ou avant), pour chaque colonne."""
    sub = df.loc[:pd.Timestamp(date)]
    return sub.ffill().iloc[-1] if len(sub) else pd.Series(np.nan, index=df.columns)


@st.cache_data(ttl=3600, show_spinner="Simulation de 10 000 portefeuilles…")
def simuler(tx, univ, debut, fin, n=10_000, graine=42):
    """Tire n portefeuilles au hasard avec tes règles (nombre de lignes, poids, limites secteur et place)
    et les compare à tes lignes détenues au début de la période, conservées jusqu'à la fin."""
    debut, fin = pd.Timestamp(debut), pd.Timestamp(fin)
    pos0, cash0, _, _, _, _ = traiter_journal(tx[tx["date"] <= debut])
    if pos0.empty:
        return {"erreur": "Aucune ligne détenue à la date de début : choisis une date après tes premiers achats."}
    tick = tuple(sorted(set(univ.index) | set(pos0.index)))
    dev = {t: devise_de(t) for t in tick}
    pr = sorted({("EURGBP=X" if d == "GBp" else f"EUR{d}=X") for d in dev.values() if d != "EUR"})
    cl, ad = historique(tuple(sorted(set(tick) | set(pr) | {BENCH_PRINCIPAL})))
    cl, ad = en_euros(cl, dev), en_euros(ad, dev)
    hist = ad.loc[:debut].iloc[-253:]
    r_hist = hist.pct_change(fill_method=None).iloc[1:]
    p0, p1 = prix_au(ad, debut), prix_au(ad, fin)
    ok = [t for t in tick if t in r_hist.columns and r_hist[t].notna().sum() > 180 and pd.notna(p0.get(t)) and pd.notna(p1.get(t))]
    exclus_moi = [t for t in pos0.index if t not in ok]
    rend = (p1[ok] / p0[ok] - 1).values
    R = r_hist[ok].fillna(0)
    cov = R.cov().values * 252
    lam = 0.94
    poids_exp = lam ** np.arange(len(R))[::-1]
    ewma = np.sqrt((poids_exp[:, None] * R.values ** 2).sum(0) / poids_exp.sum() * 252)
    # --- ton portefeuille au début (cours de clôture en euros)
    c0 = prix_au(cl, debut)
    val = pd.Series({t: pos0.loc[t, "quantite"] * c0.get(t, np.nan) for t in pos0.index}).dropna()
    total0 = val.sum() + cash0
    w_moi = np.array([val.get(t, 0.0) / total0 for t in ok])
    investi = float(w_moi.sum())
    n_lignes = int((w_moi > 0).sum())
    # --- tirages, uniquement dans l'univers choisi
    idx_u = [i for i, t in enumerate(ok) if t in univ.index]
    sect = univ["secteur"].reindex(ok).fillna("?").values
    place = univ["place"].reindex(ok).fillna("?").values
    s_code = pd.factorize(sect)[0]
    p_code = pd.factorize(place)[0]
    wb = np.where(ewma > LIM["vol_reduite"], POIDS_REDUIT, POIDS_BASE)
    rng = np.random.default_rng(graine)
    W = np.zeros((n, len(ok)))
    incomplets = 0
    for k in range(n):
        st_ = np.zeros(s_code.max() + 1)
        pt_ = np.zeros(p_code.max() + 1)
        cnt = 0
        for i in rng.permutation(idx_u):
            w = wb[i]
            if st_[s_code[i]] + w > LIM["secteur"] + 1e-9 or pt_[p_code[i]] + w > LIM["place"] + 1e-9:
                continue
            W[k, i] = w
            st_[s_code[i]] += w
            pt_[p_code[i]] += w
            cnt += 1
            if cnt == n_lignes:
                break
        if cnt < n_lignes:
            incomplets += 1
    W = W / W.sum(1, keepdims=True) * investi
    r_alea = W @ rend
    v_alea = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    r_moi = float(w_moi @ rend)
    v_moi = float(np.sqrt(w_moi @ cov @ w_moi))
    b = p1.get(BENCH_PRINCIPAL, np.nan) / p0.get(BENCH_PRINCIPAL, np.nan) - 1
    return {"r_alea": r_alea, "v_alea": v_alea, "r_moi": r_moi, "v_moi": v_moi, "n_lignes": n_lignes,
            "investi": investi, "univers_n": len(idx_u), "exclus_moi": exclus_moi, "incomplets": incomplets,
            "bench": b, "debut": debut, "fin": fin}


# ================================================================== PAGE
st.session_state["_derniere_maj"] = dt.datetime.now().timestamp()


@st.fragment(run_every=60)
def actualisation_auto():
    """Relance toute la page chaque minute pour reprendre les cours (si l'option est cochée)."""
    if st.session_state.get("auto", True) and dt.datetime.now().timestamp() - st.session_state.get("_derniere_maj", 0) > 55:
        st.rerun()


attente = st.empty()
attente.info("Chargement du portefeuille et des cours… (jusqu'à une minute au premier lancement)")
log("lecture du journal")
tx, lignes, source = charger_tables()
log(f"journal lu : {len(tx)} lignes ({source})")
positions, cash, apports, ventes, erreurs_journal, instantanes = traiter_journal(tx)

tickers_tous = sorted(set(positions.index) | set(t for t in tx["ticker"] if t))
devises = {t: devise_de(t, lignes["devise"].get(t) if t in lignes.index else None) for t in tickers_tous}
paires = sorted({("EURGBP=X" if d == "GBp" else f"EUR{d}=X") for d in devises.values() if d != "EUR"})
univers = tuple(sorted(set(tickers_tous) | set(BENCH) | set(paires)))

close, adj = historique(univers)
attente.empty()
if close.empty:
    st.error("Impossible de récupérer les cours chez Yahoo Finance pour l'instant. Réessaie dans quelques minutes.")
    st.stop()
close_eur, adj_eur = en_euros(close, devises), en_euros(adj, devises)

with st.sidebar:
    st.header("NextWise")
    st.caption(f"Positions : {source}")
    st.caption("Cours : Yahoo Finance, différés d'environ 15 minutes, convertis en euros.")
    if MOCK:
        st.warning("Mode test : cours simulés.")
    st.toggle("Actualiser chaque minute", value=True, key="auto")
    actualisation_auto()
    if st.button("Rafraîchir les cours maintenant"):
        intraday.clear()
        charger_tables.clear()
        st.rerun()
    st.caption("Les chiffres officiels du concours restent ceux de NextWise.")


def instantane_direct():
    """Cours du moment, variation du jour et valorisation des positions."""
    intra = intraday(univers)
    aujourdhui = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    intra_eur = en_euros(intra.ffill(), devises) if not intra.empty else pd.DataFrame()
    dernier, veille, heure = {}, {}, {}
    for t in tickers_tous + list(BENCH):
        if t not in close_eur:
            continue
        s = close_eur[t].dropna()
        avant = s[s.index < aujourdhui]
        live = intra_eur[t].dropna() if (not intra_eur.empty and t in intra_eur) else pd.Series(dtype=float)
        live = live[live.index.normalize().tz_localize(None) == aujourdhui] if not live.empty else live
        if not live.empty:
            dernier[t], heure[t] = live.iloc[-1], live.index[-1]
            veille[t] = avant.iloc[-1] if len(avant) else np.nan
        elif len(s):
            dernier[t], heure[t] = s.iloc[-1], s.index[-1]
            veille[t] = s.iloc[-2] if len(s) > 1 else np.nan
    pf = positions.copy()
    pf["cours"] = [dernier.get(t, np.nan) for t in pf.index]
    pf["veille"] = [veille.get(t, np.nan) for t in pf.index]
    pf["var_jour"] = pf["cours"] / pf["veille"] - 1
    pf["valeur"] = pf["quantite"] * pf["cours"]
    pf["gain_jour"] = pf["quantite"] * (pf["cours"] - pf["veille"])
    pf["latent"] = pf["quantite"] * (pf["cours"] - pf["pru"])
    pf["latent_pct"] = pf["cours"] / pf["pru"] - 1
    titres = pf["valeur"].sum(skipna=True)
    total = titres + cash
    pf["poids"] = pf["valeur"] / total
    for c in ["nom", "poche", "secteur", "place", "poids_cible", "these", "menace", "sortie", "evenement"]:
        pf[c] = [lignes[c].get(t) if t in lignes.index else None for t in pf.index]
    pf["nom"] = [n if isinstance(n, str) and n else t for t, n in zip(pf.index, pf["nom"])]
    return pf, total, titres, intra_eur, dernier, veille, heure


pf, total, titres, intra_eur, dernier, veille, heure = instantane_direct()
stats = stats_lignes(adj_eur[[c for c in adj_eur.columns if not c.endswith("=X")]], BENCH_PRINCIPAL)
pf = pf.join(stats, how="left")
poids = pf["poids"].dropna()
risque = risque_portefeuille(poids, adj_eur, BENCH_PRINCIPAL)

# ---------------------------------------------------------------- alertes
alertes = []
for t, r in pf.iterrows():
    if pd.isna(r["cours"]):
        alertes.append(("bad", f"{r['nom']} : aucun cours trouvé chez Yahoo pour « {t} ». Vérifie le ticker."))
        continue
    if r["var_jour"] < LIM["alerte_jour"]:
        alertes.append(("bad", f"{r['nom']} perd {pct(r['var_jour'], signe=True)} aujourd'hui : relis sa condition de sortie."))
    if pd.notna(r.get("momentum")) and r["momentum"] < LIM["momentum"]:
        alertes.append(("bad", f"{r['nom']} : momentum 12-1 à {pct(r['momentum'], signe=True)}, sous ta limite de {pct(LIM['momentum'], 0)}."))
    if pd.notna(r.get("vol_ewma")) and r["vol_ewma"] > LIM["vol_reduite"] and r["poids"] > 0.03:
        alertes.append(("warn", f"{r['nom']} : vol EWMA {pct(r['vol_ewma'], 0)}, poids {pct(r['poids'])} au-dessus des 2,5 % prévus."))
    if t not in lignes.index or not isinstance(lignes.loc[t, "these"], str) or not lignes.loc[t, "these"].strip():
        alertes.append(("warn", f"{r['nom']} : fiche à compléter (thèse, menace, condition de sortie) dans l'onglet « lignes »."))
cons = contraintes(pf.dropna(subset=["poids"]))
for _, c in cons.iterrows():
    if c["etat"] == "Dépassée":
        alertes.append(("bad", f"Contrainte dépassée : {c['regle']} à {pct(c['niveau'])} pour {pct(c['limite'], 0)}."))
    elif c["etat"] == "Proche":
        alertes.append(("warn", f"Contrainte proche : {c['regle']} à {pct(c['niveau'])} pour {pct(c['limite'], 0)}."))
for e in erreurs_journal:
    alertes.append(("warn", "Journal : " + e))

# ---------------------------------------------------------------- en-tête
st.title("Portefeuille NextWise")
perf = total / apports - 1 if apports else np.nan
gain_jour = pf["gain_jour"].sum(skipna=True)
var_jour_ptf = gain_jour / (total - gain_jour) if total else np.nan
bench_jour = dernier.get(BENCH_PRINCIPAL, np.nan) / veille.get(BENCH_PRINCIPAL, np.nan) - 1 if veille.get(BENCH_PRINCIPAL) else np.nan
h = max(heure.values()) if heure else None
st.caption(
    f"Mis à jour à {pd.Timestamp(h).strftime('%H:%M') if h is not None and getattr(pd.Timestamp(h), 'hour', 0) else 'la dernière clôture'}"
    f" · {len(pf)} lignes · {len([a for a in alertes if a[0] == 'bad'])} alertes rouges"
)
k = st.columns(5)
k[0].metric("Valeur totale", eur(total), f"{pct(var_jour_ptf, 2, True)} aujourd'hui")
k[1].metric("Gain du jour", eur(gain_jour), f"Euronext 100 {pct(bench_jour, 2, True)}", delta_color="off")
k[2].metric("Depuis le départ", pct(perf, 2, True), eur(total - apports))
k[3].metric("Cash", eur(cash), f"{pct(cash / total)} du total", delta_color="off")
if risque:
    k[4].metric("Volatilité prévue", pct(risque["vol"]), f"bêta {risque['beta']:.2f}".replace(".", ","), delta_color="off")

onglets = st.tabs(["Direct", "Lignes", "Performance", "Risque", "Simulateur", "Stress tests", "Chance ou talent", "Journal", "Données"])

# ================================================================== DIRECT
with onglets[0]:
    g, d = st.columns([3, 2])
    with g:
        st.subheader("Carte du portefeuille")
        st.caption("Taille : poids dans le portefeuille. Couleur : variation du jour.")
        carte = pf.dropna(subset=["valeur"]).copy()
        carte["poche"] = carte["poche"].fillna("Sans poche")
        carte["etiquette"] = carte["nom"] + "<br>" + carte["var_jour"].map(lambda v: pct(v, 1, True))
        fig = px.treemap(carte.reset_index(), path=["poche", "etiquette"], values="valeur", color="var_jour",
                         color_continuous_scale=["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#1a9850"][::1],
                         range_color=(-0.03, 0.03), color_continuous_midpoint=0,
                         custom_data=["nom", "poids", "latent_pct"])
        fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Poids %{customdata[1]:.1%}<br>Latent %{customdata[2]:+.1%}<extra></extra>",
                          texttemplate="%{label}", textfont_size=13)
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=430, coloraxis_showscale=False)
        st.plotly_chart(fig, key="carte")
    with d:
        st.subheader("Alertes")
        if not alertes:
            st.success("Rien à signaler.")
        for niveau, texte in sorted(alertes, key=lambda a: a[0] != "bad")[:12]:
            (st.error if niveau == "bad" else st.warning)(texte)
        st.subheader("Contraintes")
        st.dataframe(
            cons.style.format({"niveau": lambda v: pct(v), "limite": lambda v: "≤ " + pct(v, 0)}).map(couleur_etat, subset=["etat"]),
            hide_index=True, column_config={"regle": "Règle", "niveau": "Niveau", "limite": "Limite", "etat": "État"})

    st.subheader("Aujourd'hui")
    if not intra_eur.empty:
        ok = [t for t in pf.index if t in intra_eur.columns]
        val = (intra_eur[ok].ffill() * pf.loc[ok, "quantite"]).sum(axis=1) + cash
        base = (pf.loc[ok, "quantite"] * pf.loc[ok, "veille"]).sum() + cash
        courbe = val / base - 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=courbe.index, y=courbe, name="Portefeuille", line=dict(width=2)))
        if BENCH_PRINCIPAL in intra_eur and veille.get(BENCH_PRINCIPAL):
            b = intra_eur[BENCH_PRINCIPAL].ffill() / veille[BENCH_PRINCIPAL] - 1
            fig.add_trace(go.Scatter(x=b.index, y=b, name="Euronext 100", line=dict(width=1.5, dash="dot", color=C_NEUTRE)))
        fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
        fig.update_layout(height=280, margin=dict(t=10, l=0, r=0, b=0), yaxis_tickformat="+.1%", legend=dict(orientation="h", y=1.1), hovermode="x unified")
        st.plotly_chart(fig, key="jour")
    else:
        st.caption("Marché fermé : pas de cours intrajournaliers. Les variations affichées sont celles de la dernière séance.")

    st.subheader("Plus fortes variations du jour")
    tri = pf.dropna(subset=["var_jour"]).sort_values("var_jour")
    fig = go.Figure(go.Bar(x=tri["var_jour"], y=tri["nom"], orientation="h",
                           marker_color=[C_BON if v >= 0 else C_MAUVAIS for v in tri["var_jour"]],
                           text=[pct(v, 1, True) for v in tri["var_jour"]], textposition="outside",
                           hovertemplate="%{y} : %{x:+.2%}<extra></extra>"))
    fig.update_layout(height=max(300, 22 * len(tri)), margin=dict(t=10, l=0, r=40, b=0), xaxis_tickformat="+.0%")
    st.plotly_chart(fig, key="varjour")

# ================================================================== LIGNES
with onglets[1]:
    st.subheader("Toutes les lignes")
    poches = ["Toutes"] + sorted(pf["poche"].dropna().unique().tolist())
    choix = st.selectbox("Poche", poches)
    vue = pf if choix == "Toutes" else pf[pf["poche"] == choix]
    table = vue[["nom", "poche", "quantite", "pru", "cours", "var_jour", "valeur", "poids", "latent", "latent_pct",
                 "momentum", "vol_ewma", "beta", "haut52"]].copy()
    if risque:
        table["part_risque"] = risque["contrib"].reindex(table.index)
    st.dataframe(
        table.sort_values("poids", ascending=False).style.format({
            "quantite": "{:g}", "pru": lambda v: eur(v, 2), "cours": lambda v: eur(v, 2), "var_jour": lambda v: pct(v, 2, True),
            "valeur": lambda v: eur(v), "poids": lambda v: pct(v), "latent": lambda v: eur(v), "latent_pct": lambda v: pct(v, 1, True),
            "momentum": lambda v: pct(v, 0, True), "vol_ewma": lambda v: pct(v, 0), "beta": lambda v: "—" if pd.isna(v) else f"{v:.2f}".replace(".", ","),
            "haut52": lambda v: pct(v, 0), "part_risque": lambda v: pct(v)})
        .map(signe_couleur, subset=["var_jour", "latent", "latent_pct", "momentum"]),
        height=min(38 * (len(table) + 1), 900),
        column_config={"nom": "Ligne", "poche": "Poche", "quantite": "Qté", "pru": "Prix de revient", "cours": "Cours",
                       "var_jour": "Jour", "valeur": "Valeur", "poids": "Poids", "latent": "Latent €", "latent_pct": "Latent %",
                       "momentum": "Momentum 12-1", "vol_ewma": "Vol EWMA", "beta": "Bêta", "haut52": "% du plus haut 52 s.",
                       "part_risque": "Part du risque"})

    st.subheader("Fiche d'une ligne")
    noms = pf.sort_values("nom")["nom"]
    sel = st.selectbox("Ligne", noms.index, format_func=lambda t: pf.loc[t, "nom"])
    r = pf.loc[sel]
    a, b = st.columns([3, 2])
    with a:
        s = close_eur[sel].dropna().iloc[-260:] if sel in close_eur else pd.Series(dtype=float)
        if len(s):
            fig = go.Figure(go.Scatter(x=s.index, y=s, line=dict(width=2), name="Cours (€)"))
            fig.add_hline(y=r["pru"], line_dash="dash", line_color=C_ATTENTION, annotation_text="prix de revient", annotation_position="top left")
            fig.add_hline(y=s.max(), line_dash="dot", line_color=C_NEUTRE, annotation_text="plus haut 52 semaines", annotation_position="bottom left")
            fig.update_layout(height=320, margin=dict(t=10, l=0, r=0, b=0), yaxis_title="€")
            st.plotly_chart(fig, key="fiche")
    with b:
        st.markdown(f"### {r['nom']}")
        st.caption(f"{sel} · {r['poche'] or 'sans poche'} · {r['secteur'] or 'secteur ?'} · {r['place'] or 'place ?'}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Poids", pct(r["poids"]), f"cible {pct(r['poids_cible'])}" if pd.notna(r["poids_cible"]) else None, delta_color="off")
        m2.metric("Latent", pct(r["latent_pct"], 1, True), eur(r["latent"]))
        m3.metric("Momentum 12-1", pct(r.get("momentum"), 0, True))
        for titre, champ in [("Thèse", "these"), ("Menace", "menace"), ("Je sors si", "sortie"), ("Prochain événement", "evenement")]:
            v = r[champ]
            st.markdown(f"**{titre}** : {v if isinstance(v, str) and v.strip() else '*à compléter*'}")

# ================================================================== PERFORMANCE
with onglets[2]:
    st.subheader("Valeur du portefeuille depuis le départ")
    debut = tx["date"].dropna().min()
    jours = close_eur.index[close_eur.index >= debut]
    if len(jours):
        qte = pd.DataFrame(0.0, index=jours, columns=tickers_tous)
        cash_j = pd.Series(0.0, index=jours)
        apport_j = pd.Series(0.0, index=jours)
        for _, rr in tx.iterrows():
            if pd.isna(rr["date"]):
                continue
            m = jours >= rr["date"]
            typ, t, q, p, f = rr["type"], rr["ticker"], rr["quantite"], rr["prix"], rr["frais"]
            if typ == "apport":
                cash_j[m] += p; apport_j[m] += p
            elif typ == "retrait":
                cash_j[m] -= p; apport_j[m] -= p
            elif typ == "achat" and t:
                qte.loc[m, t] += q; cash_j[m] -= q * p + f
            elif typ == "vente" and t:
                qte.loc[m, t] -= q; cash_j[m] += q * p - f
            elif typ == "frais":
                cash_j[m] -= f if f else (p if pd.notna(p) else 0)
            elif typ == "dividende":
                cash_j[m] += ((q * p) if pd.notna(q) and pd.notna(p) else (p if pd.notna(p) else 0)) - f
        px_j = close_eur.reindex(jours)[tickers_tous].ffill()
        valeur_j = (qte * px_j).sum(axis=1, min_count=1).fillna(0) + cash_j
        # aujourd'hui : valeur en direct
        valeur_j.iloc[-1] = total if jours[-1] >= pd.Timestamp.now(tz=TZ).normalize().tz_localize(None) else valeur_j.iloc[-1]
        flux = apport_j.diff().fillna(apport_j.iloc[0])
        rend = (valeur_j - flux) / valeur_j.shift(1) - 1
        rend.iloc[0] = valeur_j.iloc[0] / apport_j.iloc[0] - 1 if apport_j.iloc[0] else 0
        indice = (1 + rend.fillna(0)).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indice.index, y=indice - 1, name="Portefeuille", line=dict(width=2.5)))
        for bt, bn in BENCH.items():
            if bt in close_eur:
                bs = close_eur[bt].reindex(jours).ffill()
                fig.add_trace(go.Scatter(x=bs.index, y=bs / bs.iloc[0] - 1, name=bn, line=dict(width=1.5, dash="dot")))
        fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
        fig.update_layout(height=380, margin=dict(t=10, l=0, r=0, b=0), yaxis_tickformat="+.1%", hovermode="x unified", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, key="perf")
        c1, c2, c3 = st.columns(3)
        dd = indice / indice.cummax() - 1
        c1.metric("Plus forte baisse depuis un sommet", pct(dd.min(), 1))
        jr = rend.dropna().iloc[1:]
        c2.metric("Volatilité réalisée (annualisée)", pct(jr.std() * np.sqrt(252)) if len(jr) > 5 else "—")
        c3.metric("Ratio de Sharpe réalisé", f"{jr.mean() / jr.std() * np.sqrt(252):.2f}".replace(".", ",") if len(jr) > 60 and jr.std() > 0 else "dès 3 mois d’historique")
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=C_MAUVAIS, width=1)))
        fig.update_layout(height=180, margin=dict(t=10, l=0, r=0, b=0), yaxis_tickformat=".1%", title=dict(text="Baisse depuis le dernier sommet", font_size=13))
        st.plotly_chart(fig, key="dd")

    st.subheader("Qui a fait la performance")
    st.caption("Plus-value latente par ligne, plus les ventes déjà réalisées.")
    contrib = pf["latent"].dropna().rename("latent").to_frame()
    contrib["nom"] = pf["nom"]
    if not ventes.empty:
        real = ventes.groupby("ticker")["resultat"].sum()
        for t, v in real.items():
            if t in contrib.index:
                contrib.loc[t, "latent"] += v
            else:
                contrib.loc[t] = [v, lignes["nom"].get(t, t) if t in lignes.index else t]
    contrib = contrib.sort_values("latent")
    fig = go.Figure(go.Bar(x=contrib["latent"], y=contrib["nom"], orientation="h",
                           marker_color=[C_BON if v >= 0 else C_MAUVAIS for v in contrib["latent"]],
                           text=[eur(v) for v in contrib["latent"]], textposition="outside"))
    fig.update_layout(height=max(300, 22 * len(contrib)), margin=dict(t=10, l=0, r=60, b=0))
    st.plotly_chart(fig, key="contrib")

    st.subheader("Par poche")
    gp = pf.groupby(pf["poche"].fillna("Sans poche")).agg(valeur=("valeur", "sum"), latent=("latent", "sum"), cout=("pru", lambda s: (s * pf.loc[s.index, "quantite"]).sum()))
    gp["poids"] = gp["valeur"] / total
    gp["rendement"] = gp["latent"] / gp["cout"]
    st.dataframe(gp[["poids", "valeur", "latent", "rendement"]].sort_values("poids", ascending=False).style.format(
        {"poids": lambda v: pct(v), "valeur": lambda v: eur(v), "latent": lambda v: eur(v), "rendement": lambda v: pct(v, 1, True)}).map(signe_couleur, subset=["latent", "rendement"]),
        column_config={"poche": "Poche", "poids": "Poids", "valeur": "Valeur", "latent": "Latent", "rendement": "Rendement"})

# ================================================================== RISQUE
with onglets[3]:
    if not risque:
        st.info("Pas assez d'historique pour calculer le risque.")
    else:
        st.caption(f"Sur un an de cours quotidiens en euros. Lignes couvertes : {pct(risque['couverture'], 0)} du capital investi.")
        c = st.columns(5)
        c[0].metric("Volatilité prévue", pct(risque["vol"]))
        c[1].metric("Ratio de diversification", f"{risque['dr']:.2f}".replace(".", ","))
        c[2].metric("Bêta (Euronext 100)", f"{risque['beta']:.2f}".replace(".", ","))
        c[3].metric("VaR 95 % sur un jour", eur(risque["var95"] * total), pct(-risque["var95"], 2), delta_color="off")
        c[4].metric("Perte moyenne au-delà (ES)", eur(risque["es95"] * total), pct(-risque["es95"], 2), delta_color="off")

        st.subheader("Capital contre risque")
        st.caption("À droite de la diagonale, la ligne pèse plus dans le risque que dans le capital.")
        cr = pd.DataFrame({"poids": poids, "risque": risque["contrib"]}).dropna()
        cr["nom"] = pf["nom"]
        cr["secteur"] = pf["secteur"].fillna("Non renseigné")
        fig = px.scatter(cr.reset_index(), x="poids", y="risque", text="nom", color="secteur", hover_name="nom")
        mx = max(cr["poids"].max(), cr["risque"].max()) * 1.1
        fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dot", color=C_NEUTRE))
        fig.update_traces(textposition="top center", textfont_size=10, marker_size=10)
        fig.update_layout(height=460, xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="Part du capital", yaxis_title="Part du risque", margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, key="capr")

        st.subheader("Secteurs : capital et risque")
        sec = pd.DataFrame({"capital": poids.groupby(pf["secteur"].fillna("Non renseigné")).sum(),
                            "risque": risque["contrib"].groupby(pf["secteur"].reindex(risque["contrib"].index).fillna("Non renseigné")).sum()}).fillna(0).sort_values("capital")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=sec.index, x=sec["capital"], name="Capital", orientation="h"))
        fig.add_trace(go.Bar(y=sec.index, x=sec["risque"], name="Risque", orientation="h"))
        fig.add_vline(x=LIM["secteur"], line_dash="dash", line_color=C_MAUVAIS, annotation_text="limite")
        fig.update_layout(barmode="group", height=max(300, 45 * len(sec)), xaxis_tickformat=".0%", margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, key="sect")

        st.subheader("Corrélations entre les lignes")
        st.caption("Rangées par ressemblance : les blocs foncés sont des lignes qui bougent ensemble.")
        corr = risque["corr"]
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform
            dist = squareform((1 - corr.fillna(0).values).clip(0), checks=False)
            ordre = corr.index[leaves_list(linkage(dist, "average"))]
            corr = corr.loc[ordre, ordre]
        except Exception:
            pass
        noms_c = [pf["nom"].get(t, t) for t in corr.index]
        fig = px.imshow(corr.values, x=noms_c, y=noms_c, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=620, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, key="corr")
        m = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        if len(m):
            (a1, a2), v = m.idxmax(), m.max()
            st.caption(f"Paire la plus corrélée : {pf['nom'].get(a1, a1)} et {pf['nom'].get(a2, a2)} ({v:.2f}).".replace(".", ",", 1))

# ================================================================== SIMULATEUR
with onglets[4]:
    st.subheader("Tester un ordre avant de le passer")
    st.caption("Modifie les poids, ajoute une ligne avec son ticker Yahoo (ex. ENEL.MI) ou mets un poids à 0 pour une vente. Le reste va en cash.")
    base = pf[["nom", "poids", "secteur", "place"]].copy()
    base["poids"] = (base["poids"] * 100).round(2)
    base = base.reset_index().rename(columns={"index": "ticker"})
    edit = st.data_editor(base, num_rows="dynamic", key="simu", hide_index=True,
                          column_config={"ticker": "Ticker Yahoo", "nom": "Nom", "poids": st.column_config.NumberColumn("Poids (%)", min_value=0.0, max_value=100.0, step=0.1),
                                         "secteur": "Secteur", "place": "Place"})
    edit = edit.dropna(subset=["ticker"])
    edit["ticker"] = edit["ticker"].astype(str).str.strip().str.upper()
    edit = edit[edit["ticker"] != ""]
    edit["nom"] = [n if isinstance(n, str) and n else t for t, n in zip(edit["ticker"], edit["nom"])]
    edit["poids"] = edit["poids"].fillna(0) / 100
    nouveaux = tuple(sorted(set(edit["ticker"]) - set(adj_eur.columns)))
    adj_sim = adj_eur
    if nouveaux:
        dev_n = {t: devise_de(t) for t in nouveaux}
        pn = sorted({("EURGBP=X" if d == "GBp" else f"EUR{d}=X") for d in dev_n.values() if d != "EUR"})
        _, adj_n = historique(tuple(sorted(set(nouveaux) | set(pn))))
        if not adj_n.empty:
            adj_n = en_euros(adj_n, dev_n)
            adj_sim = adj_eur.join(adj_n[[c for c in adj_n.columns if c in nouveaux]], how="left")
        manquants = [t for t in nouveaux if t not in adj_sim.columns or adj_sim[t].notna().sum() < 120]
        if manquants:
            st.warning("Pas assez d'historique Yahoo pour : " + ", ".join(manquants) + ". Vérifie le ticker.")
    w_sim = edit.set_index("ticker")["poids"]
    w_sim = w_sim[w_sim > 0]
    if w_sim.sum() > 1:
        st.error(f"Le total fait {pct(w_sim.sum())} : il dépasse 100 %.")
    else:
        apres = risque_portefeuille(w_sim, adj_sim, BENCH_PRINCIPAL)
        if apres and risque:
            c = st.columns(5)
            c[0].metric("Volatilité", pct(apres["vol"]), pct(apres["vol"] - risque["vol"], 2, True), delta_color="inverse")
            c[1].metric("Ratio de diversification", f"{apres['dr']:.2f}".replace(".", ","), f"{apres['dr'] - risque['dr']:+.2f}".replace(".", ","))
            c[2].metric("Bêta", f"{apres['beta']:.2f}".replace(".", ","), f"{apres['beta'] - risque['beta']:+.2f}".replace(".", ","), delta_color="off")
            c[3].metric("VaR 95 % sur un jour", pct(apres["var95"], 2), pct(apres["var95"] - risque["var95"], 2, True), delta_color="inverse")
            c[4].metric("Cash", pct(1 - w_sim.sum()), pct((1 - w_sim.sum()) - cash / total, 1, True), delta_color="off")
            cs = contraintes(edit[edit["poids"] > 0].set_index("ticker"))
            st.dataframe(cs.style.format({"niveau": lambda v: pct(v), "limite": lambda v: "≤ " + pct(v, 0)}).map(couleur_etat, subset=["etat"]),
                         hide_index=True, column_config={"regle": "Règle après l'ordre", "niveau": "Niveau", "limite": "Limite", "etat": "État"})
            s_av, s_ap = stress(poids, adj_sim, BENCH_PRINCIPAL), stress(w_sim, adj_sim, BENCH_PRINCIPAL)
            if not s_av.empty and not s_ap.empty:
                comp = s_av[["scenario", "portefeuille"]].merge(s_ap[["scenario", "portefeuille"]], on="scenario", suffixes=("_avant", "_apres"))
                comp["ecart"] = comp["portefeuille_apres"] - comp["portefeuille_avant"]
                st.dataframe(comp.style.format({c: (lambda v: pct(v, 1, True)) for c in ["portefeuille_avant", "portefeuille_apres", "ecart"]}).map(signe_couleur, subset=["ecart"]),
                             hide_index=True, column_config={"scenario": "Crise rejouée", "portefeuille_avant": "Avant", "portefeuille_apres": "Après", "ecart": "Écart"})

# ================================================================== STRESS
with onglets[5]:
    st.subheader("Si une crise passée se répétait")
    st.caption("Les poids d'aujourd'hui appliqués aux vrais cours de chaque crise. Une ligne qui n'existait pas encore reçoit le rendement moyen des autres.")
    st_df = stress(poids, adj_eur, BENCH_PRINCIPAL)
    if st_df.empty:
        st.info("Pas assez d'historique.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=st_df["scenario"], x=st_df["portefeuille"], name="Portefeuille", orientation="h", text=[pct(v, 1, True) for v in st_df["portefeuille"]], textposition="outside"))
        fig.add_trace(go.Bar(y=st_df["scenario"], x=st_df["reference"], name="Euronext 100", orientation="h", marker_color=C_NEUTRE))
        fig.update_layout(barmode="group", height=460, xaxis_tickformat=".0%", margin=dict(t=10, l=0, r=40, b=0), legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, key="stress")
        st.dataframe(st_df.style.format({"portefeuille": lambda v: pct(v, 1, True), "reference": lambda v: pct(v, 1, True), "couverture": lambda v: pct(v, 0)}),
                     hide_index=True, column_config={"scenario": "Crise", "portefeuille": "Portefeuille", "reference": "Euronext 100", "couverture": "Capital avec historique"})
    st.subheader("Choc de marché instantané")
    choc = st.slider("Baisse de l'Euronext 100", -40, 0, -10, format="%d %%")
    if risque and pd.notna(risque["beta"]):
        perte = risque["beta"] * choc / 100
        st.metric("Perte estimée du portefeuille", pct(perte, 1, True), eur(perte * total))
        st.caption("Estimation par le bêta : elle suppose que les corrélations restent normales, ce qui sous-estime la perte dans une vraie crise.")

# ================================================================== CHANCE OU TALENT
with onglets[6]:
    st.subheader("Chance ou talent ?")
    st.caption("On tire 10 000 portefeuilles au hasard qui suivent tes règles : même nombre de lignes, poids de 4,14 % "
               "(2,5 % si la vol EWMA dépasse 40 %), au plus 20 % par secteur et 30 % par place, même part investie. "
               "On les compare à tes lignes détenues au début de la période, conservées jusqu'à la fin.")
    achats = tx.loc[tx["type"] == "achat", "date"].dropna()
    aujourdhui_d = dt.datetime.now(TZ).date()
    defaut = achats.value_counts().idxmax().date() if len(achats) else aujourdhui_d  # le jour où tu as passé le plus d'ordres
    c1, c2, c3 = st.columns([1, 1, 2])
    d_debut = c1.date_input("Début", value=min(defaut, aujourdhui_d), max_value=aujourdhui_d, format="DD/MM/YYYY")
    d_fin = c2.date_input("Fin", value=aujourdhui_d, max_value=aujourdhui_d, format="DD/MM/YYYY")
    choix_u = c3.radio("Univers de tirage", ["Ma liste de départ", "Univers large"], horizontal=True,
                       help="Ma liste : les titres que tu avais présélectionnés. Univers large : environ 135 grandes valeurs des places européennes, plus exigeant.")
    if st.button("Lancer les 10 000 portefeuilles", type="primary"):
        st.session_state["simu_cle"] = (choix_u, d_debut, d_fin)
    cle = st.session_state.get("simu_cle")
    if not cle:
        st.info("Choisis la période et l'univers, puis lance la simulation.")
    elif cle[2] <= cle[1]:
        st.error("La date de fin doit être après la date de début.")
    else:
        univ = charger_univers()
        if cle[0] == "Ma liste de départ":
            univ = univ[univ["univers"] == "liste"]
        res = simuler(tx, univ, cle[1], cle[2])
        if "erreur" in res:
            st.error(res["erreur"])
        else:
            ra, va, rm, vm = res["r_alea"], res["v_alea"], res["r_moi"], res["v_moi"]
            bande = 0.01
            proches = np.abs(va - vm) <= bande
            if proches.sum() < 300:
                bande = 0.02
                proches = np.abs(va - vm) <= bande
            pct_tous = (ra < rm).mean()
            pct_risque = (ra[proches] < rm).mean() if proches.any() else np.nan
            p_val = (ra[proches] >= rm).mean() if proches.any() else np.nan
            st.caption(f"Du {res['debut']:%d/%m/%Y} au {res['fin']:%d/%m/%Y} · {cle[0].lower()} ({res['univers_n']} titres) · "
                       f"{res['n_lignes']} lignes · {pct(res['investi'], 0)} investi")
            m = st.columns(4)
            m[0].metric("Ton rendement", pct(rm, 2, True), f"Euronext 100 {pct(res['bench'], 2, True)}", delta_color="off")
            m[1].metric("Mieux que", pct(pct_tous, 0), "de tous les portefeuilles aléatoires", delta_color="off")
            m[2].metric("À risque égal, mieux que", pct(pct_risque, 0), f"{int(proches.sum())} portefeuilles à ± {pct(bande, 0)} de vol", delta_color="off")
            m[3].metric("p-valeur", f"{p_val:.2f}".replace(".", ","), "le hasard fait aussi bien dans cette part des cas", delta_color="off")
            if p_val < 0.05:
                st.success("Sur cette période, le hasard explique difficilement ton résultat. Une seule période ne prouve pas un talent : refais le test chaque mois.")
            elif p_val > 0.5:
                st.warning("Sur cette période, des portefeuilles tirés au hasard avec tes règles font aussi bien ou mieux dans la majorité des cas.")
            else:
                st.info("Résultat compatible avec le hasard : ni preuve de talent, ni preuve du contraire. C'est normal sur une période courte.")

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ra, nbinsx=80, name="Tous", marker_color="#c7c7c7", opacity=0.8))
            fig.add_trace(go.Histogram(x=ra[proches], nbinsx=80, name="À risque égal", marker_color="#4c78a8", opacity=0.8))
            fig.add_vline(x=rm, line_color=C_MAUVAIS, line_width=2, annotation_text="toi", annotation_position="top")
            if pd.notna(res["bench"]):
                fig.add_vline(x=res["bench"], line_color=C_NEUTRE, line_dash="dot", annotation_text="Euronext 100", annotation_position="bottom right")
            fig.update_layout(barmode="overlay", height=360, xaxis_tickformat="+.1%", margin=dict(t=30, l=0, r=0, b=0),
                              legend=dict(orientation="h", y=1.12), yaxis_title="nombre de portefeuilles")
            st.plotly_chart(fig, key="alea_hist")

            st.subheader("Rendement contre risque prévu")
            st.caption("Chaque point gris est un portefeuille aléatoire ; le risque est la volatilité prévue au début de la période (un an de cours).")
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=va, y=ra, mode="markers", marker=dict(size=3, color="#9e9e9e", opacity=0.35), name="Aléatoires",
                                       hovertemplate="vol %{x:.1%} · rendement %{y:+.2%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[vm], y=[rm], mode="markers+text", marker=dict(size=16, color=C_MAUVAIS, symbol="star"),
                                     text=["toi"], textposition="top center", name="Ton portefeuille"))
            fig.update_layout(height=420, xaxis_tickformat=".0%", yaxis_tickformat="+.1%", xaxis_title="volatilité prévue",
                              yaxis_title="rendement sur la période", margin=dict(t=10, l=0, r=0, b=0), showlegend=False)
            st.plotly_chart(fig, key="alea_nuage")
            notes = []
            if res["exclus_moi"]:
                notes.append("Lignes sans assez d'historique, comptées comme du cash : " + ", ".join(res["exclus_moi"]) + ".")
            if res["incomplets"]:
                notes.append(f"{res['incomplets']} tirages n'ont pas pu atteindre {res['n_lignes']} lignes à cause des limites ; leurs poids ont été remis à l'échelle.")
            notes.append("Ton portefeuille dépasse peut-être lui-même une limite (secteur à 20 %) : les tirages, eux, la respectent toujours.")
            for nt in notes:
                st.caption(nt)

# ================================================================== JOURNAL
with onglets[7]:
    st.subheader("Journal des transactions")
    st.caption("Ajoute chaque ordre exécuté dans l'onglet « transactions » de ton Google Sheet : le site le prend en compte dans la minute.")
    st.dataframe(tx.assign(date=tx["date"].dt.strftime("%d/%m/%Y")), hide_index=True,
                 column_config={"date": "Date", "ticker": "Ticker", "type": "Type", "quantite": "Quantité", "prix": "Prix", "frais": "Frais", "commentaire": "Commentaire"})
    st.subheader("Ventes réalisées")
    if ventes.empty:
        st.caption("Aucune vente pour l'instant.")
    else:
        st.dataframe(ventes.assign(date=ventes["date"].dt.strftime("%d/%m/%Y")).style.format(
            {"pru": lambda v: eur(v, 2), "prix": lambda v: eur(v, 2), "resultat": lambda v: eur(v), "resultat_pct": lambda v: pct(v, 1, True)}).map(signe_couleur, subset=["resultat", "resultat_pct"]),
            hide_index=True)
    st.subheader("Effet de disposition (Odean, 1998)")
    st.caption("À chaque vente, on compte les gains et les pertes réalisés, et ceux que tu as gardés sans vendre. Si la part de gains réalisés (PGR) dépasse nettement celle des pertes réalisées (PLR), tu coupes tes gagnants trop tôt.")
    rg = pg = rl = pl = 0
    for (d, t_vendu, autres), (_, v) in zip(instantanes, ventes.iterrows() if not ventes.empty else []):
        if v["resultat"] > 0:
            rg += 1
        else:
            rl += 1
        for t, (q, pru) in autres.items():
            prix_d = close_eur[t].loc[:d].dropna() if t in close_eur else pd.Series(dtype=float)
            if len(prix_d):
                if prix_d.iloc[-1] > pru:
                    pg += 1
                else:
                    pl += 1
    if rg + rl == 0:
        st.caption("Le calcul démarre à ta première vente.")
    else:
        pgr = rg / (rg + pg) if rg + pg else np.nan
        plr = rl / (rl + pl) if rl + pl else np.nan
        c = st.columns(3)
        c[0].metric("PGR (gains réalisés)", pct(pgr, 0))
        c[1].metric("PLR (pertes réalisées)", pct(plr, 0))
        c[2].metric("PGR − PLR", pct(pgr - plr, 0, True), "biais présent" if pgr - plr > 0.05 else "pas de biais net", delta_color="off")

# ================================================================== DONNÉES
with onglets[8]:
    st.subheader("Contrôle de la qualité des données")
    rows = []
    aujourdhui = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    for t in tickers_tous + list(BENCH):
        s = close[t].dropna() if t in close else pd.Series(dtype=float)
        if s.empty:
            rows.append({"ticker": t, "dernier_jour": "—", "jours": 0, "saut_max_30j": np.nan, "etat": "Aucun cours"})
            continue
        r30 = s.pct_change(fill_method=None).iloc[-30:].abs().max()
        retard = np.busday_count(s.index[-1].date(), aujourdhui.date())
        etat = "Cours ancien" if retard > 3 else ("Saut suspect" if r30 > 0.20 else ("Historique court" if len(s) < 250 else "OK"))
        rows.append({"ticker": t, "dernier_jour": s.index[-1].strftime("%d/%m/%Y"), "jours": len(s), "saut_max_30j": r30, "etat": etat})
    q = pd.DataFrame(rows)
    st.dataframe(q.style.format({"saut_max_30j": lambda v: pct(v, 1)}).map(lambda v: "" if v == "OK" else f"color:{C_ATTENTION};font-weight:600", subset=["etat"]),
                 hide_index=True, column_config={"ticker": "Ticker", "dernier_jour": "Dernier cours", "jours": "Jours d'historique", "saut_max_30j": "Plus fort mouvement sur 30 jours", "etat": "État"})
    st.caption("Un « saut suspect » peut être une vraie nouvelle (résultats) ou une erreur d'ajustement de Yahoo : compare avec NextWise.")
    st.caption("Devises converties en euros : " + (", ".join(f"{t} ({d})" for t, d in devises.items() if d != "EUR") or "aucune"))
