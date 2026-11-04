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
import html
import os
import urllib.parse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import analyses as A

print("[nextwise] imports terminés", flush=True)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

st.set_page_config(page_title="NextWise · suivi en direct", page_icon="📈", layout="wide")

# Réparation automatique : si le navigateur n'a pas réussi à charger un élément (graphique, bouton…), on re-télécharge
# le fichier manquant puis on recharge la page, sur la même page de l'appli. Au plus 3 fois de suite.
WATCHDOG = """
<script>
(function () {
  var P = window.parent, D;
  try { D = P.document; } catch (e) { return; }
  var CLE = "nw_reparations";
  function nb() { try { return parseInt(P.sessionStorage.getItem(CLE) || "0", 10); } catch (e) { return 9; } }
  function scan() {
    var t = D.body ? D.body.innerText : "";
    var m = t.match(/Failed to fetch dynamically imported module:\\s*(\\S+?\\.js)/);
    if (!m) return;
    if (nb() >= 3) return;
    try { P.sessionStorage.setItem(CLE, String(nb() + 1)); } catch (e) {}
    clearInterval(boucle);
    var url = m[1], essais = 0;
    function essai() {
      fetch(url, { cache: "reload" }).then(function (r) { if (!r.ok) throw 0; P.location.reload(); })
        .catch(function () { essais++; if (essais < 6) setTimeout(essai, 700); else P.location.reload(); });
    }
    essai();
  }
  var boucle = setInterval(scan, 1200);
  setTimeout(function () { try { P.sessionStorage.setItem(CLE, "0"); } catch (e) {} }, 45000);
})();
</script>
"""
components.html(WATCHDOG, height=0)


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
    for c in ["ticker", "nom", "poche", "secteur", "place", "poids_cible", "these", "menace", "sortie", "evenement", "devise",
              "type_valo", "per", "croissance", "peg", "lynch", "roe", "dette", "moat", "binaire", "diagnostic", "verdict"]:
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
    for c in ["poids_cible", "per", "croissance", "peg", "lynch", "roe", "dette"]:
        li[c] = li[c].map(num)
    li["binaire"] = pd.to_datetime(li["binaire"], format="%Y-%m-%d", errors="coerce")
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
    if os.environ.get("NEXTWISE_DEGRADE") == "1":   # test : tickers manquants, historique court, marché fermé
        df = df.drop(columns=[c for c in ["EQNR.OL", "BZ=F", "IEGA.AS", "EUNH.DE", "IBGS.AS", "EURUSD=X", "^STOXX", "ASM.AS", "OR.PA"] if c in df.columns])
        if "KRZ.IR" in df.columns:
            df.loc[df.index[:-150], "KRZ.IR"] = np.nan
    return df, df.copy()


@st.cache_data(ttl=3600, show_spinner="Téléchargement de l'historique des cours…")
def historique(tickers):
    tickers = sorted(set(tickers))
    if MOCK or yf is None:
        return _faux_historique(tickers)
    log(f"téléchargement historique : {len(tickers)} tickers")
    t0 = dt.datetime.now()
    try:
        data = yf.download(tickers, period="10y", interval="1d", auto_adjust=False, progress=False, threads=False, timeout=20)
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
    if os.environ.get("NEXTWISE_DEGRADE") == "1":
        return pd.DataFrame()
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


TABLE_CSS = """<style>
.nwt {overflow-x:auto; margin-bottom:0.5rem}
.nwt table {border-collapse:collapse; font-size:0.86rem; width:100%; font-variant-numeric:tabular-nums}
.nwt th {text-align:left; padding:6px 10px; border-bottom:1px solid rgba(128,128,128,.45); white-space:nowrap; font-weight:600}
.nwt td {padding:5px 10px; border-bottom:1px solid rgba(128,128,128,.18); white-space:nowrap}
.nwt tbody tr:hover td, .nwt tbody tr:hover th {background:rgba(128,128,128,.10)}
</style>"""


def tableau(data, column_config=None, hide_index=None, height=None, **_):
    """Tableau HTML statique : il s'affiche sans composant chargé à la demande par le navigateur,
    donc il ne peut pas « échouer ». Accepte les mêmes arguments que les appels st.dataframe d'origine."""
    sty = data if hasattr(data, "relabel_index") and hasattr(data, "to_html") else data.style.format(na_rep="")
    noms = []
    for c in sty.data.columns:
        lab = (column_config or {}).get(c, c)
        noms.append(lab if isinstance(lab, str) else str(c))
    sty = sty.relabel_index(noms, axis="columns")
    sty = sty.hide(axis="index")   # l'index n'est jamais affiché : une colonne qui compte doit être une vraie colonne
    st.html(TABLE_CSS + '<div class="nwt">' + sty.to_html() + "</div>")


def couleur_etat(v):
    return {"Dépassée": f"color:{C_MAUVAIS};font-weight:600", "Proche": f"color:{C_ATTENTION};font-weight:600",
            "Respectée": f"color:{C_BON}"}.get(v, "")


def signe_couleur(v):
    if pd.isna(v):
        return ""
    if abs(v) < 5e-5:
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
    # --- étape 1 bis : tableau de survie mois par mois (Tarnavskyi, 2020)
    survie = A.tableau_survie(W, ad[ok], ad[BENCH_PRINCIPAL], w_moi, debut, fin) if BENCH_PRINCIPAL in ad else pd.DataFrame()
    # --- portefeuilles jumeaux (Daniel et al., 1997 ; Heyman et al., 2014)
    h = ad.loc[:debut]
    mom = {}
    for t in ok:
        s_ = h[t].dropna()
        mom[t] = s_.iloc[-22] / s_.iloc[-253] - 1 if len(s_) > 253 else np.nan
    car = pd.DataFrame({"secteur": sect, "mom": pd.Series(mom).reindex(ok).values, "vol": ewma}, index=ok)
    car["mom_q"] = pd.qcut(car["mom"].rank(method="first"), 3, labels=False).fillna(-1)
    car["vol_q"] = pd.qcut(car["vol"].rank(method="first"), 3, labels=False)
    w_s = pd.Series(w_moi, index=ok)
    w_s = w_s[w_s > 0]
    garde = sorted(set(car.index[idx_u]) | set(w_s.index))
    jum = A.jumeaux(car.loc[garde], w_s, pd.Series(rend, index=ok), n=n)
    return {"r_alea": r_alea, "v_alea": v_alea, "r_moi": r_moi, "v_moi": v_moi, "n_lignes": n_lignes,
            "investi": investi, "univers_n": len(idx_u), "exclus_moi": exclus_moi, "incomplets": incomplets,
            "bench": b, "debut": debut, "fin": fin, "survie": survie, "jumeaux": jum}


# ================================================================== DONNÉES YAHOO À LA DEMANDE
FICHIER_CAL = "/tmp/nextwise_calendrier.json"


@st.cache_data(ttl=24 * 3600, show_spinner="Fondamentaux Yahoo (une à deux minutes la première fois)…")
def fondamentaux_yahoo(tickers):
    """ROE, dette nette / Ebitda, PER, croissance et PEG depuis Yahoo, à titre indicatif (à vérifier sur MarketScreener)."""
    rows = {}
    rng = np.random.default_rng(11)
    for t in tickers:
        if MOCK or yf is None:
            rows[t] = {"roe": rng.uniform(0.03, 0.35), "dette": rng.uniform(-0.5, 4.5), "per": rng.uniform(7, 35),
                       "croissance": rng.uniform(-0.2, 0.4), "peg": rng.uniform(0.5, 3.5), "capitalisation": rng.uniform(2e9, 2e11)}
            continue
        try:
            i = yf.Ticker(t).get_info()
        except Exception as e:
            log(f"fondamentaux {t} : {type(e).__name__}")
            rows[t] = {}
            continue
        dette, cash_, ebitda = i.get("totalDebt"), i.get("totalCash"), i.get("ebitda")
        nd = (dette - (cash_ or 0)) / ebitda if dette is not None and ebitda and ebitda > 0 else np.nan
        rows[t] = {"roe": i.get("returnOnEquity"), "dette": nd, "per": i.get("forwardPE") or i.get("trailingPE"),
                   "croissance": i.get("earningsGrowth"), "peg": i.get("trailingPegRatio") or i.get("pegRatio"),
                   "capitalisation": i.get("marketCap")}
    return pd.DataFrame(rows).T.apply(pd.to_numeric, errors="coerce")


def charger_calendrier(tickers):
    """Prochaines dates de résultats (Yahoo), gardées un jour dans un fichier du serveur."""
    import json
    out = {}
    for t in tickers:
        d = None
        if MOCK or yf is None:
            d = (pd.Timestamp.now() + pd.Timedelta(days=int(np.random.default_rng(abs(hash(t)) % 1000).integers(1, 90)))).strftime("%Y-%m-%d")
        else:
            try:
                cal = yf.Ticker(t).calendar
                dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
                if dates:
                    d = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
            except Exception as e:
                log(f"calendrier {t} : {type(e).__name__}")
        out[t] = d
    try:
        with open(FICHIER_CAL, "w") as f:
            json.dump({"maj": dt.datetime.now().isoformat(), "dates": out}, f)
    except Exception:
        pass
    return out


def lire_calendrier():
    import json
    try:
        with open(FICHIER_CAL) as f:
            d = json.load(f)
        if (dt.datetime.now() - dt.datetime.fromisoformat(d["maj"])).total_seconds() < 24 * 3600:
            return d["dates"], d["maj"]
    except Exception:
        pass
    return None, None


def impact_candidats(w, tick_c, R, mode, a_par_ticker, ligne_remplacee=None):
    """Volatilité et ratio de diversification du portefeuille si chaque candidat entre, soit à son poids cible financé
    par tout le portefeuille, soit à la place d'une ligne. w : poids actuels (Series), R : rendements quotidiens sur un an."""
    T = list(dict.fromkeys([t for t in w.index if t in R.columns] + [c for c in tick_c if c in R.columns]))
    cov = R[T].fillna(0).cov().values * 252
    pos = {t: i for i, t in enumerate(T)}
    wv = np.zeros(len(T))
    for t, x in w.items():
        if t in pos:
            wv[pos[t]] = x
    sig = np.sqrt(np.diag(cov))
    base_var = float(wv @ cov @ wv)
    out = {}
    for c in tick_c:
        if c not in pos:
            continue
        j = pos[c]
        if mode == "ajout":
            a = a_par_ticker.get(c, POIDS_BASE)
            w2 = (1 - a) * wv
            w2[j] += a
        else:
            w2 = wv.copy()
            r = pos.get(ligne_remplacee)
            if r is None:
                continue
            a = wv[r]
            w2[r] = 0.0
            w2[j] += a
        v = float(w2 @ cov @ w2)
        out[c] = {"vol_apres": np.sqrt(v), "dr_apres": float((w2 * sig).sum() / np.sqrt(v)) if v > 0 else np.nan}
    return pd.DataFrame(out).T, np.sqrt(base_var), float((wv * sig).sum() / np.sqrt(base_var)) if base_var > 0 else np.nan


# ================================================================== PAGE
st.session_state["_derniere_maj"] = dt.datetime.now().timestamp()


@st.fragment(run_every=60)
def actualisation_auto():
    """Relance toute la page chaque minute pour reprendre les cours (si l'option est cochée)."""
    if st.session_state.get("auto", True) and st.session_state.get("page", "Tableau de bord") == "Tableau de bord" and dt.datetime.now().timestamp() - st.session_state.get("_derniere_maj", 0) > 55:
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
    st.checkbox("Actualiser chaque minute (tableau de bord)", value=True, key="auto")
    actualisation_auto()
    st.number_input("Taux sans risque (%)", 0.0, 10.0, 2.0, 0.25, key="rf",
                    help="Taux de la facilité de dépôt de la BCE, utilisé pour les ratios de Sharpe, Sortino, Treynor et M². Vérifie-le sur le site de la BCE.")
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
    pf["cours_absent"] = pf["cours"].isna()
    # sans cours Yahoo : valorisé au prix de revient pour que la valeur totale et les poids restent justes
    pf.loc[pf["cours_absent"], "cours"] = pf.loc[pf["cours_absent"], "pru"]
    pf.loc[pf["cours_absent"], "veille"] = pf.loc[pf["cours_absent"], "pru"]
    pf["var_jour"] = pf["cours"] / pf["veille"] - 1
    pf["valeur"] = pf["quantite"] * pf["cours"]
    pf["gain_jour"] = pf["quantite"] * (pf["cours"] - pf["veille"])
    pf["latent"] = pf["quantite"] * (pf["cours"] - pf["pru"])
    pf["latent_pct"] = pf["cours"] / pf["pru"] - 1
    titres = pf["valeur"].sum(skipna=True)
    total = titres + cash
    pf["poids"] = pf["valeur"] / total
    for c in ["nom", "poche", "secteur", "place", "poids_cible", "these", "menace", "sortie", "evenement",
              "type_valo", "per", "croissance", "peg", "lynch", "roe", "dette", "moat", "binaire", "diagnostic", "verdict"]:
        pf[c] = [lignes[c].get(t) if t in lignes.index else None for t in pf.index]
    pf["nom"] = [n if isinstance(n, str) and n else t for t, n in zip(pf.index, pf["nom"])]
    return pf, total, titres, intra_eur, dernier, veille, heure


pf, total, titres, intra_eur, dernier, veille, heure = instantane_direct()
stats = stats_lignes(adj_eur[[c for c in adj_eur.columns if not c.endswith("=X")]], BENCH_PRINCIPAL)
pf = pf.join(stats, how="left")
poids = pf["poids"].dropna()
risque = risque_portefeuille(poids, adj_eur, BENCH_PRINCIPAL)

# ---------------------------------------------------------------- alertes
REGLES_TEXTE = [
    ("Entrée", "Momentum 12-1 mois", "positif"),
    ("Entrée", "Plus forte hausse sur un jour le mois dernier (effet loterie)", "8 % au plus"),
    ("Entrée", "ROE / dette nette sur Ebitda", "10 % au moins / 3 au plus (hors finance et utilities)"),
    ("Entrée", "Valorisation", "PEG ≤ 2 (croissance) ou ratio de Lynch ≥ 1 (croissance lente) ; cycliques et financières jugées à part"),
    ("Entrée", "Corrélation avec chaque autre ligne", "moins de 0,7"),
    ("Entrée", "Thèse, menace et condition de sortie", "écrites avant l'achat"),
    ("Lignes risquées", "Vol EWMA au-dessus de 40 % : poids", "2,5 %"),
    ("Lignes risquées", "Vol EWMA au-dessus de 40 % : conditions", "moat identifié, ROE ≥ 15 %, momentum positif, pas d'événement binaire avant le prochain rééquilibrage (sinon poids ≤ 1,5 %)"),
    ("Taille", "Poids de base", "98 % ÷ nombre de lignes, 4,5 % au plus à l'achat"),
    ("Portefeuille", "UCITS 5/10/40", "toujours respecté ; rééquilibrage anticipé si les lignes au-dessus de 5 % dépassent 35 %"),
    ("Portefeuille", "Secteur", "20 % du capital et 30 % du risque au plus"),
    ("Portefeuille", "Place de cotation", "30 % au plus"),
    ("Portefeuille", "Bêta", "entre 0,6 et 0,85"),
    ("Portefeuille", "Volatilité prévue", "18 % au plus, sinon cash jusqu'à 10 %"),
    ("Sortie", "Condition de sortie atteinte", "vente dans les 2 jours de bourse"),
    ("Sortie", "Momentum sous −20 %, cas 3, contrainte non respectée", "vente au rééquilibrage mensuel"),
    ("Sortie", "Prix d'achat", "n'entre dans aucune décision (effet de disposition)"),
    ("Discipline", "Exception", "écrite dans le journal avant d'agir, avec une date de réexamen"),
    ("Discipline", "Test de la page blanche", "chaque mois : mes règles me la feraient-elles acheter aujourd'hui ?"),
]


def verifier_regles(pf, corr):
    """Vérifie chaque ligne contre les règles écrites. ✓ respectée, ✗ non respectée, – sans objet."""
    ok, ko, na = "✓", "✗", "–"
    aujourdhui = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    prochain_reeq = (aujourdhui + pd.offsets.MonthBegin(1))
    rows = []
    for t, r in pf.iterrows():
        typ = r.get("type_valo") or ""
        sect = r.get("secteur") or ""
        mom, roe, dette, peg, lynch = r.get("momentum"), r.get("roe"), r.get("dette"), r.get("peg"), r.get("lynch")
        def test(cond):
            return na if cond is None else (ok if cond else ko)
        valo = None
        if typ == "Croissance" and pd.notna(peg):
            valo = peg <= 2
        elif typ == "Croissance lente" and pd.notna(lynch):
            valo = lynch >= 1
        cmax = np.nan
        if corr is not None and t in corr.index:
            cc = corr.loc[t].drop(t, errors="ignore")
            cmax = cc.max() if len(cc) else np.nan
        risquee = pd.notna(r.get("vol_ewma")) and r["vol_ewma"] > LIM["vol_reduite"]
        manque = []
        if risquee:
            if not (isinstance(r.get("moat"), str) and r["moat"].strip()):
                manque.append("pas de moat identifié")
            if not (pd.notna(roe) and roe >= 0.15):
                manque.append("ROE sous 15 %")
            if not (pd.notna(mom) and mom > 0):
                manque.append("momentum négatif")
            b = r.get("binaire")
            if pd.notna(b) and aujourdhui <= b <= prochain_reeq + pd.Timedelta(days=31) and r["poids"] > 0.015:
                manque.append(f"événement binaire le {pd.Timestamp(b):%d/%m} avec un poids de {pct(r['poids'])}")
        rows.append({
            "ticker": t, "nom": r["nom"],
            "momentum": test(None if pd.isna(mom) else mom > 0),
            "loterie": test(None if pd.isna(r.get("max_jour")) else r["max_jour"] <= 0.08),
            "roe": test(None if pd.isna(roe) else roe >= 0.10),
            "dette": test(None if (typ == "Financière" or sect == "Utilities" or pd.isna(dette)) else dette <= 3),
            "valorisation": test(valo),
            "correlation": test(None if pd.isna(cmax) else cmax < 0.7),
            "taille": test(None if pd.isna(r["poids"]) else r["poids"] <= (0.025 + 0.005 if risquee else 0.05)),
            "ligne_risquee": (ko if manque else ok) if risquee else na,
            "detail_risquee": " ; ".join(manque),
            "corr_max": cmax,
        })
    out = pd.DataFrame(rows).set_index("ticker")
    out["ecarts"] = (out[["momentum", "loterie", "roe", "dette", "valorisation", "correlation", "taille", "ligne_risquee"]] == ko).sum(axis=1)
    return out


regles_lignes = verifier_regles(pf, risque["corr"] if risque else None)

alertes = []
for t, r in pf.iterrows():
    if r.get("cours_absent", False):
        alertes.append(("bad", f"{r['nom']} : aucun cours trouvé chez Yahoo pour « {t} » ; la ligne est valorisée à son prix de revient. Vérifie le ticker dans l'onglet Données."))
        continue
    if r["var_jour"] < LIM["alerte_jour"]:
        alertes.append(("bad", f"{r['nom']} perd {pct(r['var_jour'], signe=True)} aujourd'hui : relis sa condition de sortie."))
    if pd.notna(r.get("momentum")) and r["momentum"] < LIM["momentum"]:
        alertes.append(("bad", f"{r['nom']} : momentum 12-1 à {pct(r['momentum'], signe=True)}, sous ta limite de {pct(LIM['momentum'], 0)}."))
    if pd.notna(r.get("vol_ewma")) and r["vol_ewma"] > LIM["vol_reduite"] and r["poids"] > 0.03:
        alertes.append(("warn", f"{r['nom']} : vol EWMA {pct(r['vol_ewma'], 0)}, poids {pct(r['poids'])} au-dessus des 2,5 % prévus."))
    if t not in lignes.index or not isinstance(lignes.loc[t, "these"], str) or not lignes.loc[t, "these"].strip():
        alertes.append(("warn", f"{r['nom']} : fiche à compléter (thèse, menace, condition de sortie) dans l'onglet « lignes »."))
cal_dates, cal_maj = lire_calendrier()
if cal_dates:
    auj_ = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    for t, d in cal_dates.items():
        if d and t in pf.index:
            jj = (pd.Timestamp(d) - auj_).days
            if 0 <= jj <= 3:
                alertes.append(("warn", f"{pf.loc[t, 'nom']} publie ses résultats {'aujourd’hui' if jj == 0 else f'dans {jj} jour' + ('s' if jj > 1 else '')} ({pd.Timestamp(d):%d/%m}) : relis sa condition de sortie avant."))
for t, rg in regles_lignes.iterrows():
    if rg["ligne_risquee"] == "✗":
        alertes.append(("warn", f"{rg['nom']} : ligne risquée qui ne remplit pas la règle ({rg['detail_risquee']})."))
    if rg["roe"] == "✗":
        alertes.append(("warn", f"{rg['nom']} : ROE sous 10 %, contraire à ta règle de qualité."))
cons = contraintes(pf.dropna(subset=["poids"]))
for _, c in cons.iterrows():
    if c["etat"] == "Dépassée":
        alertes.append(("bad", f"Contrainte dépassée : {c['regle']} à {pct(c['niveau'])} pour {pct(c['limite'], 0)}."))
    elif c["etat"] == "Proche":
        alertes.append(("warn", f"Contrainte proche : {c['regle']} à {pct(c['niveau'])} pour {pct(c['limite'], 0)}."))
for e in erreurs_journal:
    alertes.append(("warn", "Journal : " + e))

# ---------------------------------------------------------------- en-tête
perf = total / apports - 1 if apports else np.nan
gain_jour = pf["gain_jour"].sum(skipna=True)
var_jour_ptf = gain_jour / (total - gain_jour) if total else np.nan
bench_jour = dernier.get(BENCH_PRINCIPAL, np.nan) / veille.get(BENCH_PRINCIPAL, np.nan) - 1 if veille.get(BENCH_PRINCIPAL) else np.nan
h_maj = max(heure.values()) if heure else None
h_txt = pd.Timestamp(h_maj).strftime("%H:%M") if h_maj is not None and getattr(pd.Timestamp(h_maj), "hour", 0) else "dernière clôture"

CSS_GLOBAL = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, p, li, label, h1, h2, h3, h4, td, th, .stMarkdown, .stCaption {font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;}
[data-testid="stIconMaterial"], .material-symbols-rounded {font-family: 'Material Symbols Rounded' !important;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -0.01em;}
h2 {font-weight: 650 !important;}
h3 {font-weight: 600 !important; margin-top: 0.6rem !important;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-weight: 600;}
[data-testid="stMetricLabel"] p {font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; opacity: .75;}
.nw-top {display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:6px;}
.nw-top .titre {font-size:1.35rem; font-weight:700; letter-spacing:-.01em;}
.nw-top .etat {font-size:.8rem; opacity:.75;}
.nw-chip {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.75rem; font-weight:600; margin-left:6px;}
.nw-chip.rouge {background:rgba(239,68,68,.15); color:#ef4444;}
.nw-chip.orange {background:rgba(245,158,11,.15); color:#d97706;}
.nw-chip.vert {background:rgba(34,197,94,.15); color:#16a34a;}
.nw-chip.gris {background:rgba(127,127,127,.15);}
.nw-band {display:grid; grid-template-columns:repeat(auto-fit, minmax(132px, 1fr)); gap:10px; margin:2px 0 16px;}
.nw-k {background:rgba(127,127,127,.07); border:1px solid rgba(127,127,127,.18); border-radius:12px; padding:10px 14px;}
.nw-k .t {font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; opacity:.65;}
.nw-k .v {font-size:1.4rem; font-weight:650; font-variant-numeric:tabular-nums; line-height:1.3; margin-top:2px; white-space:nowrap;}
.nw-k .s {font-size:.76rem; opacity:.85; font-variant-numeric:tabular-nums; line-height:1.35;}
.nw-pos {color:#16a34a;} .nw-neg {color:#ef4444;} .nw-neu {opacity:.75;}
.nw-j {margin:0 0 10px;}
.nw-j .l {display:flex; justify-content:space-between; font-size:.82rem; margin-bottom:3px; gap:8px;}
.nw-j .l b {font-variant-numeric:tabular-nums;}
.nw-j .bar {height:8px; border-radius:6px; background:rgba(127,127,127,.18); position:relative; overflow:hidden;}
.nw-j .bar span {position:absolute; left:0; top:0; bottom:0; border-radius:6px;}
.nw-j .ok span {background:#22c55e;} .nw-j .pr span {background:#f59e0b;} .nw-j .ko span {background:#ef4444;}
.nw-k.solo {margin-bottom:10px;} .nw-k.solo .v {font-size:1.25rem;}
.nw-intro {font-size:.9rem; opacity:.8; margin:-4px 0 12px; max-width: 1100px;}
</style>"""


def _cl(v):
    return "nw-pos" if pd.notna(v) and v > 0 else ("nw-neg" if pd.notna(v) and v < 0 else "nw-neu")


def bandeau():
    """Le bandeau des chiffres clés, en HTML simple (il ne dépend d'aucun composant chargé à la demande)."""
    n_rouge = len([a for a in alertes if a[0] == "bad"])
    n_orange = len([a for a in alertes if a[0] == "warn"])
    chips = (f'<span class="nw-chip rouge">{n_rouge} alerte{"s" if n_rouge > 1 else ""}</span>' if n_rouge else "") + \
            (f'<span class="nw-chip orange">{n_orange} à surveiller</span>' if n_orange else "") + \
            ('<span class="nw-chip vert">tout est en ordre</span>' if not (n_rouge or n_orange) else "") + \
            ('<span class="nw-chip gris">mode test</span>' if MOCK else "")
    vol = risque["vol"] if risque else np.nan
    beta = risque["beta"] if risque else np.nan
    dr = risque["dr"] if risque else np.nan
    cartes = [
        ("Valeur totale", eur(total), f'<span class="{_cl(var_jour_ptf)}">{pct(var_jour_ptf, 2, True)}</span> aujourd\'hui'),
        ("Gain du jour", f'<span class="{_cl(gain_jour)}">{eur(gain_jour)}</span>', f'Euronext 100 <span class="{_cl(bench_jour)}">{pct(bench_jour, 2, True)}</span>'),
        ("Depuis le départ", f'<span class="{_cl(perf)}">{pct(perf, 2, True)}</span>', f'<span class="{_cl(total - apports)}">{eur(total - apports)}</span>'),
        ("Cash", eur(cash), f"{pct(cash / total)} du total"),
        ("Volatilité prévue", pct(vol), f"bêta {beta:.2f}".replace(".", ",") if pd.notna(beta) else "—"),
        ("Diversification", f"{dr:.2f}".replace(".", ",") if pd.notna(dr) else "—", f"{len(pf)} lignes"),
    ]
    html = CSS_GLOBAL + '<div class="nw-top"><div class="titre">Portefeuille NextWise</div>' + \
           f'<div class="etat">Cours à {h_txt} · différés d\'environ 15 min {chips}</div></div><div class="nw-band">' + \
           "".join(f'<div class="nw-k"><div class="t">{t}</div><div class="v">{v}</div><div class="s">{s_}</div></div>' for t, v, s_ in cartes) + "</div>"
    st.html(html)


PALETTE = ["#4F8BFF", "#22C55E", "#F59E0B", "#EF4444", "#A855F7", "#14B8A6", "#64748B", "#EC4899"]


def graphe(fig, key=None, **kw):
    """Tous les graphiques passent par ici : même police, mêmes couleurs, fond transparent, sans barre d'outils."""
    fig.update_layout(font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12), colorway=PALETTE,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_size=12))
    fig.update_xaxes(gridcolor="rgba(127,127,127,.15)", zerolinecolor="rgba(127,127,127,.35)")
    fig.update_yaxes(gridcolor="rgba(127,127,127,.15)", zerolinecolor="rgba(127,127,127,.35)")
    if any(getattr(t, "type", "") == "bar" and getattr(t, "text", None) is not None for t in fig.data):
        fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
        m = fig.layout.margin
        fig.update_layout(margin=dict(l=m.l if m.l is not None else 0, r=max(m.r or 0, 80), t=m.t if m.t is not None else 10, b=m.b if m.b is not None else 0))
    st.plotly_chart(fig, key=key, config={"displayModeBar": False, "locale": "fr"}, **kw)


def jauges(cons):
    """Contraintes en jauges : la barre montre le niveau rapporté à la limite."""
    html = ""
    for _, c in cons.iterrows():
        cl = {"Respectée": "ok", "Proche": "pr", "Dépassée": "ko"}.get(c["etat"], "ok")
        larg = min(100, 100 * c["niveau"] / c["limite"]) if c["limite"] else 0
        html += (f'<div class="nw-j"><div class="l"><span>{c["regle"]}</span><b>{pct(c["niveau"])} / {pct(c["limite"], 0)}</b></div>'
                 f'<div class="bar {cl}"><span style="width:{larg:.0f}%"></span></div></div>')
    st.html(html)


def kpi(conteneur, titre, valeur, sous=None, delta_color="normal", **_):
    """Chiffre clé en HTML (même style que le bandeau). Le sous-titre est coloré selon son signe
    (vert si positif, rouge si négatif ; l'inverse avec delta_color="inverse" ; neutre avec "off")."""
    v = html.escape(str(valeur))
    cl = ""
    if sous is not None and delta_color != "off":
        t0 = str(sous).strip()[:1]
        pos, neg = t0 == "+", t0 in ("-", "−")
        if not any(ch in "123456789" for ch in str(sous).split(" ")[0]):
            pos = neg = False      # une variation nulle (+0,00 %) reste neutre
        if delta_color == "inverse":
            pos, neg = neg, pos
        cl = " nw-pos" if pos else (" nw-neg" if neg else "")
    s_ = f'<div class="s{cl}">{html.escape(str(sous))}</div>' if sous is not None and str(sous) != "" else ""
    conteneur.html(f'<div class="nw-k solo"><div class="t">{html.escape(str(titre))}</div><div class="v">{v}</div>{s_}</div>')


def RF():
    return float(st.session_state.get("rf", 2.0)) / 100


def intro(texte):
    st.html(f'<div class="nw-intro">{texte}</div>')


def serie_portefeuille():
    """Valeur quotidienne du portefeuille réel et rendements pondérés par le temps (les apports ne comptent pas comme gain)."""
    debut = tx["date"].dropna().min()
    jours = close_eur.index[close_eur.index >= debut]
    if not len(jours):
        return None
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
    px_j = close_eur.reindex(index=jours, columns=tickers_tous).ffill()
    # titre sans aucun cours chez Yahoo : on le valorise à son dernier prix de transaction plutôt que de le faire disparaître
    for t in px_j.columns[px_j.isna().all()]:
        dern = tx.loc[(tx["ticker"] == t) & tx["prix"].notna(), "prix"]
        if len(dern):
            px_j[t] = float(dern.iloc[-1])
    valeur_j = (qte * px_j).sum(axis=1, min_count=1).fillna(0) + cash_j
    if jours[-1] >= pd.Timestamp.now(tz=TZ).normalize().tz_localize(None):
        valeur_j.iloc[-1] = total
    flux = apport_j.diff().fillna(apport_j.iloc[0])
    rend = (valeur_j - flux) / valeur_j.shift(1) - 1
    rend.iloc[0] = valeur_j.iloc[0] / apport_j.iloc[0] - 1 if apport_j.iloc[0] else 0
    indice = (1 + rend.fillna(0)).cumprod()
    return {"jours": jours, "valeur": valeur_j, "rend": rend, "indice": indice}


# ================================================================== DIRECT
def page_direct():
    g, d = st.columns([3, 2])
    with g:
        st.subheader("Carte du portefeuille")
        st.caption("Taille : poids dans le portefeuille. Couleur : variation du jour.")
        carte = pf.dropna(subset=["valeur"]).copy()
        carte["poche"] = carte["poche"].fillna("Sans poche")
        carte["etiquette"] = carte["nom"] + "<br>" + carte["var_jour"].map(lambda v: pct(v, 1, True))
        fig = px.treemap(carte.reset_index(), path=[px.Constant("Portefeuille"), "poche", "etiquette"], values="valeur", color="var_jour",
                         color_continuous_scale=["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#1a9850"][::1],
                         range_color=(-0.03, 0.03), color_continuous_midpoint=0,
                         custom_data=["nom", "poids", "latent_pct"])
        fig.update_traces(root_color="rgba(0,0,0,0)", pathbar_visible=False, marker=dict(line=dict(width=1.5, color="rgba(255,255,255,.6)"), pad=dict(t=22, l=3, r=3, b=3)),
                          hovertemplate="<b>%{customdata[0]}</b><br>Poids %{customdata[1]:.1%}<br>Latent %{customdata[2]:+.1%}<extra></extra>",
                          texttemplate="%{label}", textfont_size=13)
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=430, coloraxis_showscale=False)
        graphe(fig, key="carte", theme=None)
    with d:
        st.subheader("Alertes")
        if not alertes:
            st.success("Rien à signaler.")
        for niveau, texte in sorted(alertes, key=lambda a: a[0] != "bad")[:12]:
            (st.error if niveau == "bad" else st.warning)(texte)
        st.subheader("Contraintes")
        jauges(cons)

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
        graphe(fig, key="jour")
    else:
        st.caption("Marché fermé : pas de cours intrajournaliers. Les variations affichées sont celles de la dernière séance.")

    st.subheader("Plus fortes variations du jour")
    tri = pf.dropna(subset=["var_jour"]).sort_values("var_jour")
    fig = go.Figure(go.Bar(x=tri["var_jour"], y=tri["nom"], orientation="h",
                           marker_color=[C_BON if v >= 0 else C_MAUVAIS for v in tri["var_jour"]],
                           text=[pct(v, 1, True) for v in tri["var_jour"]], textposition="outside",
                           hovertemplate="%{y} : %{x:+.2%}<extra></extra>"))
    fig.update_layout(height=max(300, 22 * len(tri)), margin=dict(t=10, l=0, r=40, b=0), xaxis_tickformat="+.0%")
    graphe(fig, key="varjour")

# ================================================================== LIGNES
def page_lignes():
    st.subheader("Toutes les lignes")
    poches = ["Toutes"] + sorted(pf["poche"].dropna().unique().tolist())
    choix = st.selectbox("Poche", poches)
    vue = pf if choix == "Toutes" else pf[pf["poche"] == choix]
    table = vue[["nom", "poche", "quantite", "pru", "cours", "var_jour", "valeur", "poids", "latent", "latent_pct",
                 "momentum", "vol_ewma", "beta", "haut52"]].copy()
    if risque:
        table["part_risque"] = risque["contrib"].reindex(table.index)
    tableau(
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
            graphe(fig, key="fiche")
    with b:
        st.markdown(f"### {r['nom']}")
        st.caption(f"{sel} · {r['poche'] or 'sans poche'} · {r['secteur'] or 'secteur ?'} · {r['place'] or 'place ?'}")
        m1, m2, m3 = st.columns(3)
        kpi(m1, "Poids", pct(r["poids"]), f"cible {pct(r['poids_cible'])}" if pd.notna(r["poids_cible"]) else None)
        kpi(m2, "Latent", pct(r["latent_pct"], 1, True), eur(r["latent"]))
        kpi(m3, "Momentum 12-1", pct(r.get("momentum"), 0, True))
        for titre, champ in [("Thèse", "these"), ("Menace", "menace"), ("Je sors si", "sortie"), ("Prochain événement", "evenement"), ("Moat", "moat"), ("Valorisation", "diagnostic"), ("Décision écrite", "verdict")]:
            v = r[champ]
            st.markdown(f"**{titre}** : {v if isinstance(v, str) and v.strip() else '*à compléter*'}")

    st.subheader("Prochains résultats")
    cal_d, cal_m = lire_calendrier()
    if st.button("Charger les dates de résultats (Yahoo)"):
        with st.spinner("Dates de résultats…"):
            cal_d = charger_calendrier(list(pf.index))
        cal_m = dt.datetime.now().isoformat()
    if cal_d:
        auj_ = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
        cdf = pd.DataFrame([{"nom": pf.loc[t, "nom"], "date": pd.Timestamp(d), "dans": (pd.Timestamp(d) - auj_).days}
                            for t, d in cal_d.items() if d and t in pf.index]).sort_values("date")
        cdf = cdf[cdf["dans"] >= -1]
        tableau(cdf.style.format({"date": lambda v: f"{v:%d/%m/%Y}", "dans": lambda v: "aujourd'hui" if v == 0 else (f"dans {v} j" if v > 0 else "hier")})
                .map(lambda v: f"color:{C_ATTENTION};font-weight:600" if isinstance(v, (int, np.integer)) and 0 <= v <= 7 else "", subset=["dans"]),
                hide_index=True, column_config={"nom": "Ligne", "date": "Résultats", "dans": "Échéance"})
        st.caption(f"Dates Yahoo, mises à jour le {pd.Timestamp(cal_m):%d/%m à %H:%M} et gardées 24 h. Une alerte s'affiche sur la page Direct à 3 jours. "
                   "Yahoo se trompe parfois de quelques jours : vérifie sur le site de l'entreprise pour une ligne sensible.")
    else:
        st.caption("Clique sur le bouton pour récupérer les dates (environ 20 secondes).")

# ================================================================== PERFORMANCE
def page_performance():
    st.subheader("Valeur du portefeuille depuis le départ")
    sp = serie_portefeuille()
    if sp:
        jours, rend, indice = sp["jours"], sp["rend"], sp["indice"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indice.index, y=indice - 1, name="Portefeuille", line=dict(width=2.5)))
        for bt, bn in BENCH.items():
            if bt in close_eur:
                bs = close_eur[bt].reindex(jours).ffill()
                fig.add_trace(go.Scatter(x=bs.index, y=bs / bs.iloc[0] - 1, name=bn, line=dict(width=1.5, dash="dot")))
        fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
        fig.update_layout(height=380, margin=dict(t=10, l=0, r=0, b=0), yaxis_tickformat="+.1%", hovermode="x unified", legend=dict(orientation="h", y=1.08))
        graphe(fig, key="perf")
        c1, c2, c3 = st.columns(3)
        dd = indice / indice.cummax() - 1
        kpi(c1, "Plus forte baisse depuis un sommet", pct(dd.min(), 1))
        jr = rend.dropna().iloc[1:]
        kpi(c2, "Volatilité réalisée (annualisée)", pct(jr.std() * np.sqrt(252)) if len(jr) > 5 else "—")
        sl = A.sharpe_lo(jr, RF()) if len(jr) > 60 else None
        kpi(c3, "Ratio de Sharpe (Lo, net du taux sans risque)", f"{sl['sr_lo']:.2f}".replace(".", ",") if sl and pd.notna(sl["sr_lo"]) else "après 3 mois",
            "corrigé de l'autocorrélation (Lo, 2002)", delta_color="off")
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=C_MAUVAIS, width=1)))
        fig.update_layout(height=200, margin=dict(t=36, l=0, r=0, b=0), yaxis_tickformat=".1%", title=dict(text="Baisse depuis le dernier sommet", font_size=13))
        graphe(fig, key="dd")

    st.subheader("Ratios de risque et de rendement")
    intro("Les indicateurs des logiciels de gestion (Morningstar Direct, Bloomberg PORT), pour ton portefeuille et pour l'Euronext 100, sur la même période. "
          f"Taux sans risque utilisé : {pct(RF(), 2)} (réglable dans le menu de gauche).")
    premier = tx.loc[tx["type"] == "achat", "date"].dropna()
    premier = premier.value_counts().idxmax() if len(premier) else None
    r_reel = sp["rend"].dropna() if sp else pd.Series(dtype=float)
    if premier is not None:
        r_reel = r_reel[r_reel.index > premier]
    options = ["Portefeuille actuel rejoué sur un an", "Portefeuille réel depuis le départ"]
    choix = st.radio("Série analysée", options, index=1 if len(r_reel) >= 60 else 0, horizontal=True, key="ratios_serie",
                     help="Le portefeuille réel est la vraie mesure, mais il faut quelques semaines d'historique. Le rejoué applique tes poids actuels "
                          "à l'année passée : il décrit ta stratégie, sans prouver de talent (biais de sélection).")
    r_ser = r_reel if choix == options[1] else portefeuille_rejoue(252)
    r_ind = adj_eur[BENCH_PRINCIPAL].pct_change(fill_method=None) if BENCH_PRINCIPAL in adj_eur else None
    rt = A.ratios(r_ser, r_ind, RF()) if r_ind is not None else None
    if not rt:
        st.info(f"Il faut au moins 20 séances pour ces ratios ({len(r_ser)} pour l'instant) : choisis « rejoué sur un an » en attendant.")
    else:
        tb_r, n_j = rt
        pc_ = {"Rendement annualisé", "Volatilité annualisée", "Plus forte baisse", "VaR 95 % historique (1 jour)", "VaR 95 % Cornish-Fisher (1 jour)",
               "Expected Shortfall 95 % (1 jour)", "Alpha de Jensen (annualisé)", "Écart de suivi", "M² (Modigliani)", "Jours au-dessus de l'indice",
               "R² ajusté face à l'indice", "Indice d'ulcère"}
        mieux_haut = {"Rendement annualisé", "Ratio de Sharpe (Lo)", "Ratio de Sortino", "Ratio de Calmar", "Ratio Omega", "Alpha de Jensen (annualisé)",
                      "Ratio de Treynor", "Ratio d'information", "M² (Modigliani)", "Capture haussière", "Jours au-dessus de l'indice"}
        mieux_bas = {"Volatilité annualisée", "Indice d'ulcère", "VaR 95 % historique (1 jour)", "VaR 95 % Cornish-Fisher (1 jour)", "Expected Shortfall 95 % (1 jour)",
                     "Capture baissière", "Bêta à la baisse"}

        def fmt(row, col):
            v = row[col]
            if pd.isna(v):
                return "—"
            if row["ratio"] in pc_:
                return pct(v, 1, row["ratio"] in ("Alpha de Jensen (annualisé)",))
            if row["ratio"] == "Plus forte baisse":
                return pct(v, 1)
            return f"{v:.2f}".replace(".", ",")
        vue = tb_r.copy()
        vue["verdict"] = ""
        for i, row in vue.iterrows():
            if pd.notna(row["portefeuille"]) and pd.notna(row["indice"]) and row["ratio"] not in ("Bêta", "Écart de suivi", "R² ajusté face à l'indice"):
                if row["ratio"] == "Plus forte baisse":
                    vue.at[i, "verdict"] = "mieux" if row["portefeuille"] > row["indice"] else "moins bien"
                elif row["ratio"] in mieux_haut:
                    vue.at[i, "verdict"] = "mieux" if row["portefeuille"] > row["indice"] else "moins bien"
                elif row["ratio"] in mieux_bas:
                    vue.at[i, "verdict"] = "mieux" if row["portefeuille"] < row["indice"] else "moins bien"
        vue["portefeuille_txt"] = [fmt(r_, "portefeuille") for _, r_ in vue.iterrows()]
        vue["indice_txt"] = [fmt(r_, "indice") for _, r_ in vue.iterrows()]
        tableau(vue[["ratio", "portefeuille_txt", "indice_txt", "verdict", "definition"]].style
                .map(lambda v: f"color:{C_BON};font-weight:600" if v == "mieux" else (f"color:{C_MAUVAIS};font-weight:600" if v == "moins bien" else ""), subset=["verdict"]),
                hide_index=True, column_config={"ratio": "Indicateur", "portefeuille_txt": "Portefeuille", "indice_txt": "Euronext 100", "verdict": "Face à l'indice",
                                                "definition": "Ce que ça mesure"})
        st.caption(f"Calculé sur {n_j} séances. Sur moins de six mois, ces ratios sont très instables : lis-les comme des tendances, pas comme des verdicts.")

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
    graphe(fig, key="contrib")

    st.subheader("Par poche")
    gp = pf.groupby(pf["poche"].fillna("Sans poche")).agg(valeur=("valeur", "sum"), latent=("latent", "sum"), cout=("pru", lambda s: (s * pf.loc[s.index, "quantite"]).sum()))
    gp["poids"] = gp["valeur"] / total
    gp["rendement"] = gp["latent"] / gp["cout"]
    tableau(gp[["poids", "valeur", "latent", "rendement"]].sort_values("poids", ascending=False).rename_axis("poche").reset_index().style.format(
        {"poids": lambda v: pct(v), "valeur": lambda v: eur(v), "latent": lambda v: eur(v), "rendement": lambda v: pct(v, 1, True)}).map(signe_couleur, subset=["latent", "rendement"]),
        column_config={"poche": "Poche", "poids": "Poids", "valeur": "Valeur", "latent": "Latent", "rendement": "Rendement"})


    st.markdown("### Attribution par secteur (Brinson)")
    intro("Ta performance face à un univers de référence à poids égaux (ta liste de départ), découpée en deux décisions : "
          "<b>l'allocation</b> (avoir mis plus ou moins de poids dans les bons secteurs, cash compris) et <b>la sélection</b> "
          "(avoir choisi de meilleurs titres que la moyenne du secteur). Méthode de Brinson et Fachler (1985).")
    achats_b = tx.loc[tx["type"] == "achat", "date"].dropna()
    if not len(achats_b):
        st.info("Pas encore d'achat dans le journal.")
        return
    d0_def = achats_b.value_counts().idxmax().date()
    c1, c2 = st.columns(2)
    d0 = pd.Timestamp(c1.date_input("Début", value=d0_def, format="DD/MM/YYYY", key="br_d0"))
    d1 = pd.Timestamp(c2.date_input("Fin", value=dt.datetime.now(TZ).date(), format="DD/MM/YYYY", key="br_d1"))
    if d1 <= d0:
        st.error("La date de fin doit être après la date de début.")
        return
    pos_b, cash_b, *_ = traiter_journal(tx[tx["date"] <= d0])
    pu_b, _, uu_b = prix_univers(True)
    tout_b = adj_eur.join(pu_b[[c for c in pu_b.columns if c not in adj_eur.columns]], how="outer")
    p0, p1 = prix_au(tout_b, d0), prix_au(tout_b, d1)
    c0 = prix_au(close_eur, d0)
    val0 = pd.Series({t: pos_b.loc[t, "quantite"] * c0.get(t, np.nan) for t in pos_b.index}).dropna()
    tot0 = val0.sum() + cash_b
    if tot0 <= 0 or val0.empty:
        st.info("Aucune ligne détenue à la date de début.")
        return
    w_lignes = val0 / tot0
    r_lignes = (p1 / p0 - 1).reindex(w_lignes.index).fillna(0)
    sect_l = pd.Series({t: (lignes["secteur"].get(t) if t in lignes.index else None) or "Autre" for t in w_lignes.index})
    bench_t = [t for t in uu_b.index if t in p0.index and pd.notna(p0[t]) and pd.notna(p1[t])]
    r_bench = (p1[bench_t] / p0[bench_t] - 1)
    sect_b = uu_b.loc[bench_t, "secteur"]
    tab = pd.DataFrame({"w_b": sect_b.value_counts(normalize=True), "r_b": r_bench.groupby(sect_b).mean()})
    wp = w_lignes.groupby(sect_l).sum()
    rp_s = (w_lignes * r_lignes).groupby(sect_l).sum() / wp
    tab = tab.join(pd.DataFrame({"w_p": wp, "r_p": rp_s}), how="outer").fillna({"w_b": 0, "w_p": 0})
    tab["r_b"] = tab["r_b"].fillna(tab["r_p"])
    tab.loc["Cash", ["w_p", "r_p", "w_b", "r_b"]] = [cash_b / tot0, 0.0, 0.0, 0.0]
    att, Rb = A.brinson(tab)
    Rp = float((tab["w_p"] * tab["r_p"].fillna(0)).sum())
    c = st.columns(4)
    kpi(c[0], "Ton rendement (lignes du début)", pct(Rp, 2, True))
    kpi(c[1], "Univers à poids égaux", pct(Rb, 2, True))
    kpi(c[2], "Effet allocation", pct(att["allocation"].sum(), 2, True))
    kpi(c[3], "Effet sélection (+ interaction)", pct(att["selection"].sum() + att["interaction"].sum(), 2, True))
    att = att.sort_values("total")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=att.index, x=att["allocation"], name="Allocation", orientation="h", marker_color=PALETTE[0]))
    fig.add_trace(go.Bar(y=att.index, x=att["selection"] + att["interaction"], name="Sélection", orientation="h", marker_color=PALETTE[1]))
    fig.update_layout(barmode="relative", height=max(280, 30 * len(att)), xaxis_tickformat="+.2%", margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=1.08))
    graphe(fig, key="brinson")
    tableau(att[["w_p", "w_b", "r_p", "r_b", "allocation", "selection", "total"]].sort_values("total", ascending=False).rename_axis("secteur").reset_index().style.format(
        {"w_p": lambda v: pct(v), "w_b": lambda v: pct(v), "r_p": lambda v: pct(v, 2, True), "r_b": lambda v: pct(v, 2, True),
         "allocation": lambda v: pct(v, 2, True), "selection": lambda v: pct(v, 2, True), "total": lambda v: pct(v, 2, True)})
        .map(signe_couleur, subset=["allocation", "selection", "total"]),
        column_config={"secteur": "Secteur", "w_p": "Ton poids", "w_b": "Poids univers", "r_p": "Ton rendement", "r_b": "Rendement univers",
                       "allocation": "Allocation", "selection": "Sélection", "total": "Total"})
    st.caption("La ligne « Cash » mesure ce que t'a coûté (ou rapporté) le fait de ne pas être investi à 100 %. Les lignes achetées après la date de début ne comptent pas : "
               "choisis une date de début après tes derniers achats.")

# ================================================================== RISQUE
def page_risque():
    if not risque:
        st.info("Pas assez d'historique pour calculer le risque.")
    else:
        st.caption(f"Sur un an de cours quotidiens en euros. Lignes couvertes : {pct(risque['couverture'], 0)} du capital investi.")
        c = st.columns(5)
        kpi(c[0], "Volatilité prévue", pct(risque["vol"]))
        kpi(c[1], "Ratio de diversification", f"{risque['dr']:.2f}".replace(".", ","))
        kpi(c[2], "Bêta (Euronext 100)", f"{risque['beta']:.2f}".replace(".", ","))
        kpi(c[3], "VaR 95 % sur un jour", eur(risque["var95"] * total), pct(-risque["var95"], 2))
        kpi(c[4], "Perte moyenne au-delà (ES)", eur(risque["es95"] * total), pct(-risque["es95"], 2))
        r_cf = portefeuille_rejoue(252)
        vcf = A.var_cornish_fisher(r_cf)
        nm_ = A.normalite(r_cf)
        if pd.notna(vcf) and nm_:
            c = st.columns(3)
            kpi(c[0], "VaR 95 % Cornish-Fisher", eur(vcf * total), f"{pct(-vcf, 2)} · corrigée des queues épaisses", delta_color="off")
            kpi(c[1], "Asymétrie des rendements", f"{nm_['asymetrie']:+.2f}".replace(".", ","), "négative = baisses plus violentes que les hausses", delta_color="off")
            kpi(c[2], "Kurtosis", f"{nm_['kurtosis']:.2f}".replace(".", ","), "au-dessus de 3 = queues épaisses", delta_color="off")

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
        graphe(fig, key="capr")

        st.subheader("Secteurs : capital et risque")
        sec = pd.DataFrame({"capital": poids.groupby(pf["secteur"].fillna("Non renseigné")).sum(),
                            "risque": risque["contrib"].groupby(pf["secteur"].reindex(risque["contrib"].index).fillna("Non renseigné")).sum()}).fillna(0).sort_values("capital")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=sec.index, x=sec["capital"], name="Capital", orientation="h"))
        fig.add_trace(go.Bar(y=sec.index, x=sec["risque"], name="Risque", orientation="h"))
        fig.add_vline(x=LIM["secteur"], line_dash="dash", line_color=C_MAUVAIS, annotation_text="limite")
        fig.update_layout(barmode="group", height=max(300, 45 * len(sec)), xaxis_tickformat=".0%", margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=1.05))
        graphe(fig, key="sect")

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
        graphe(fig, key="corr")
        m = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        if len(m):
            (a1, a2), v = m.idxmax(), m.max()
            st.caption(f"Paire la plus corrélée : {pf['nom'].get(a1, a1)} et {pf['nom'].get(a2, a2)} ({v:.2f}).".replace(".", ",", 1))


    st.markdown("### D'où vient ton risque : l'exposition aux facteurs")
    intro("Une régression de ton portefeuille sur les grands facteurs connus. Un bêta élevé sur un facteur veut dire que ton portefeuille "
          "se comporte en partie comme ce facteur : c'est un pari, voulu ou non. Ce qui n'est expliqué par aucun facteur est ta sélection propre.")
    rp1 = portefeuille_rejoue(504)
    ff = french_europe()
    pu_f, bu_f, _ = prix_univers(True)
    if ff is not None and not ff.empty and len(ff.index.intersection(rp1.index)) > 120:
        fac = ff.copy()
        fac.insert(0, "Marché (Euronext 100)", adj_eur[BENCH_PRINCIPAL].pct_change(fill_method=None))
        source_f = f"Fama-French Europe (Kenneth French), données jusqu'au {ff.index.max():%d/%m/%Y}, et Euronext 100 pour le marché"
    else:
        fac = pd.DataFrame({"Marché (Euronext 100)": adj_eur[BENCH_PRINCIPAL].pct_change(fill_method=None),
                            "Momentum (maison)": wml_cache(pu_f), "Basse volatilité (maison)": bv_cache(pu_f)})
        source_f = ("facteurs construits sur ta liste de départ : marché (Euronext 100), momentum (gagnants − perdants) et basse volatilité "
                    "(les moins volatils − les plus volatils, dans l'esprit de Frazzini et Pedersen, 2014)")
    fac = fac.reindex(rp1.index)
    rg = A.regression_multi(rp1, fac)
    if rg is None:
        st.info("Pas assez de données communes pour la régression.")
    else:
        st.caption(f"Portefeuille actuel rejoué sur deux ans · {source_f}.")
        c = st.columns(3)
        kpi(c[0], "Part expliquée par les facteurs (R² ajusté)", pct(rg["r2_adj"], 1), f"R² brut {pct(rg['r2'], 1)}", delta_color="off")
        kpi(c[1], "Alpha annualisé", pct(rg["alpha"] * 252, 1, True), f"t = {rg['t_alpha']:.1f}".replace(".", ","))
        kpi(c[2], "Jours utilisés", rg["n"])
        g, d = st.columns(2)
        with g:
            bt = rg["betas"]
            fig = go.Figure(go.Bar(x=bt.values, y=bt.index, orientation="h",
                                   marker_color=[PALETTE[0] if abs(rg["t"][i]) >= 2 else "#9aa4b2" for i in bt.index],
                                   text=[f"{v:+.2f} (t {rg['t'][i]:.1f})".replace(".", ",") for i, v in bt.items()], textposition="outside"))
            fig.update_layout(height=300, margin=dict(t=30, l=0, r=60, b=0), title=dict(text="Bêtas (en bleu : significatifs, |t| ≥ 2)", font_size=13))
            graphe(fig, key="fac_betas")
        with d:
            dv = A.decomposition_variance(rg)
            fig = go.Figure(go.Bar(x=dv.values, y=dv.index, orientation="h", marker_color=PALETTE[2], text=[pct(v, 0) for v in dv.values], textposition="outside"))
            fig.update_layout(height=300, xaxis_tickformat=".0%", margin=dict(t=30, l=0, r=40, b=0), title=dict(text="Part de la variance de ton portefeuille", font_size=13))
            graphe(fig, key="fac_var")
        st.caption("Lecture : un bêta momentum positif veut dire que tu détiens surtout des titres en tendance ; un bêta basse volatilité positif, que tu penches "
                   "vers les titres calmes. Une part « propre aux titres » élevée correspond à un R² faible : c'est un signe de sélection active (Amihud et Goyenko, 2013).")
        sens = A.sensibilites(adj_eur[[t for t in poids.index if t in adj_eur.columns]].pct_change(fill_method=None).reindex(fac.index), fac.dropna())
        if not sens.empty:
            sens.index = [pf["nom"].get(t, t) for t in sens.index]
            fig = px.imshow(sens.T.values, x=list(sens.index), y=list(sens.columns), color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5, aspect="auto",
                            text_auto=".2f")
            fig.update_layout(height=120 + 45 * sens.shape[1], margin=dict(t=30, l=0, r=0, b=0), title=dict(text="Bêta de chaque ligne à chaque facteur", font_size=13),
                              coloraxis_showscale=False)
            graphe(fig, key="fac_lignes")

    st.markdown("### Liquidité de tes lignes")
    intro("Combien de jours il faudrait pour vendre chaque ligne sans peser sur le cours, en ne prenant pas plus de 20 % du volume quotidien moyen "
          "(règle courante des tables de négociation). Le règlement de NextWise précise que l'exécution dépend de la liquidité : une ligne peu échangée "
          "peut être exécutée à un moins bon prix.")
    vm = volumes_moyens(tuple(pf.index))
    if vm is None or vm.empty:
        st.info("Volumes indisponibles pour l'instant.")
    else:
        lq = pd.DataFrame({"nom": pf["nom"], "quantite": pf["quantite"], "volume": vm.reindex(pf.index)})
        lq["part"] = lq["quantite"] / lq["volume"]
        lq["jours"] = lq["quantite"] / (0.2 * lq["volume"])
        lq["euros_jour"] = lq["volume"] * pf["cours"]
        lq = lq.sort_values("part", ascending=False)
        tableau(lq.style.format({"quantite": "{:g}", "volume": lambda v: "—" if pd.isna(v) else f"{v:,.0f}".replace(",", "\u202f"),
                                 "part": lambda v: "—" if pd.isna(v) else pct(v, 3), "jours": lambda v: "—" if pd.isna(v) else ("moins d'un jour" if v < 1 else f"{v:.1f} j".replace(".", ",")),
                                 "euros_jour": lambda v: "—" if pd.isna(v) else eur(v)}),
                hide_index=True, column_config={"nom": "Ligne", "quantite": "Titres détenus", "volume": "Volume moyen (3 mois)", "part": "Part du volume d'un jour",
                                                "jours": "Jours pour tout vendre", "euros_jour": "Montant échangé par jour"})
        st.caption("Avec 50 000 € répartis sur 25 lignes de grandes valeurs, tout se vend en moins d'un jour : la liquidité n'est pas un risque pour toi. "
                   "Elle le deviendrait avec des petites capitalisations ou des ETF peu échangés.")

# ================================================================== MOMENTUM
@st.cache_data(ttl=3600, show_spinner="Cours de l'univers…")
def prix_univers(liste_seule=True):
    u = charger_univers()
    if liste_seule:
        u = u[u["univers"] == "liste"]
    dev = {t: devise_de(t) for t in u.index}
    pr = sorted({("EURGBP=X" if d == "GBp" else f"EUR{d}=X") for d in dev.values() if d != "EUR"})
    _, ad = historique(tuple(sorted(set(u.index) | set(pr) | {BENCH_PRINCIPAL})))
    ad = en_euros(ad, dev)
    return ad[[c for c in ad.columns if c in u.index]], ad[BENCH_PRINCIPAL] if BENCH_PRINCIPAL in ad else None, u


@st.cache_data(ttl=3600, show_spinner="Facteur basse volatilité…")
def bv_cache(prix):
    return A.facteur_basse_vol(prix)


@st.cache_data(ttl=24 * 3600, show_spinner="Facteurs Fama-French Europe (Kenneth French)…")
def french_europe():
    """Facteurs européens quotidiens de la bibliothèque de Kenneth French (Dartmouth). Vide si le téléchargement échoue."""
    if MOCK:
        return pd.DataFrame()
    import io, urllib.request, zipfile
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    out = []
    for nom in ["Europe_5_Factors_Daily_CSV.zip", "Europe_Mom_Factor_Daily_CSV.zip"]:
        try:
            req = urllib.request.Request(base + nom, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=25).read()
            z = zipfile.ZipFile(io.BytesIO(data))
            txt = z.read(z.namelist()[0]).decode("latin-1")
            out.append(A.lire_french(txt))
        except Exception as e:
            log(f"Fama-French {nom} : {type(e).__name__}: {e}")
    if not out or any(o.empty for o in out):
        return pd.DataFrame()
    df = out[0].join(out[1], how="inner")
    mom = [c for c in df.columns if c.upper().startswith(("WML", "MOM"))]
    ren = {"SMB": "Taille (SMB)", "HML": "Value (HML)", "RMW": "Rentabilité (RMW)", "CMA": "Investissement (CMA)"}
    if mom:
        ren[mom[0]] = "Momentum (WML)"
    return df.rename(columns=ren)[[v for v in ren.values() if v in df.rename(columns=ren).columns]]


@st.cache_data(ttl=3600, show_spinner="Sensibilités aux facteurs de marché…")
def series_macro():
    """Pétrole (Brent), euro/dollar et obligations d'État européennes (ETF) pour les scénarios à curseurs."""
    cand = {"Pétrole (Brent)": ["BZ=F"], "Euro / dollar": ["EURUSD=X"], "Obligations d'État € (prix)": ["IEGA.AS", "EUNH.DE", "IBGS.AS"]}
    tout = sorted({t for l in cand.values() for t in l})
    cl, _ = historique(tuple(tout))
    out = {}
    for nom, liste in cand.items():
        for t in liste:
            if t in cl.columns and cl[t].dropna().shape[0] > 300:
                out[nom] = cl[t]
                break
    return pd.DataFrame(out)


@st.cache_data(ttl=3600, show_spinner="Coefficients d'information…")
def ic_cache(prix):
    return A.ic_mensuel(prix, 36)


@st.cache_data(ttl=3600, show_spinner="Historique des scores…")
def scores_hist_cache(prix, bench):
    return A.scores_historiques(prix, bench, 26)


@st.cache_data(ttl=6 * 3600, show_spinner="Volumes échangés…")
def volumes_moyens(tickers):
    """Volume quotidien moyen sur trois mois (nombre de titres), pour mesurer la liquidité."""
    if MOCK or yf is None:
        rng = np.random.default_rng(4)
        return pd.Series({t: float(rng.uniform(2e4, 3e6)) for t in tickers})
    try:
        data = yf.download(list(tickers), period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False, timeout=20)
        v = data["Volume"] if isinstance(data.columns, pd.MultiIndex) else data[["Volume"]].rename(columns={"Volume": tickers[0]})
        return v.mean()
    except Exception as e:
        log(f"volumes : {type(e).__name__}: {e}")
        return pd.Series(dtype=float)


def portefeuille_rejoue(jours=252):
    """Tes poids d'aujourd'hui appliqués aux rendements passés (le cash rapporte 0)."""
    r = adj_eur.pct_change(fill_method=None).iloc[-jours:]
    t = [x for x in poids.index if x in r.columns]
    return r[t].fillna(0) @ poids.reindex(t).fillna(0)


@st.cache_data(ttl=3600, show_spinner="Calcul des signaux de momentum…")
def signaux_cache(prix, bench):
    return A.signaux_momentum(prix, bench)


@st.cache_data(ttl=3600, show_spinner="Construction du facteur momentum européen…")
def wml_cache(prix):
    return A.facteur_momentum(prix)


def page_momentum():
    st.subheader("Momentum et tendance")
    st.caption("Le momentum est l'une des anomalies les plus solides de la finance, mais à un horizon précis : les gagnants des 3 à 12 derniers mois "
               "continuent de gagner (Jegadeesh et Titman, 1993 ; Rouwenhorst, 1998 en Europe), alors que sur un mois, c'est l'inverse : les plus fortes "
               "hausses ont tendance à reculer (Jegadeesh, 1990). Cette page sépare les deux.")
    lignes_p = [t for t in pf.index if t in adj_eur.columns]
    if BENCH_PRINCIPAL not in adj_eur:
        st.error("Indice de référence indisponible.")
    else:
        sg = signaux_cache(adj_eur[lignes_p], adj_eur[BENCH_PRINCIPAL])
        sg["nom"] = pf["nom"].reindex(sg.index)
        sg["poids"] = pf["poids"].reindex(sg.index)
        sg = sg.sort_values("score", ascending=False)
        st.markdown("### Tes lignes, classées par score de momentum")
        st.caption("Le score combine quatre signaux, chacun ramené à la même échelle : momentum résiduel, momentum intermédiaire, proximité du plus haut "
                   "sur 52 semaines et régularité de la hausse. Plus il est élevé, plus la tendance de la ligne est solide.")
        tb = sg[["nom", "score", "mom_resid", "mom_12_7", "mom_12_1", "id_fip", "haut52", "tendance_10m", "rend_1m", "poids"]].copy()

        def c_score(v):
            return "" if pd.isna(v) else (f"color:{C_BON};font-weight:600" if v > 0.5 else (f"color:{C_MAUVAIS};font-weight:600" if v < -0.5 else ""))

        def c_inv(v):
            return f"color:{C_ATTENTION};font-weight:600" if pd.notna(v) and v > 0.15 else ""

        tableau(tb.style.format({"score": lambda v: "—" if pd.isna(v) else f"{v:+.2f}".replace(".", ","),
                                 "mom_resid": lambda v: "—" if pd.isna(v) else f"{v:+.2f}".replace(".", ","),
                                 "mom_12_7": lambda v: pct(v, 0, True), "mom_12_1": lambda v: pct(v, 0, True),
                                 "id_fip": lambda v: "—" if pd.isna(v) else f"{v:+.2f}".replace(".", ","),
                                 "haut52": lambda v: pct(v, 0), "tendance_10m": lambda v: ("au-dessus " if v > 0 else "en dessous ") + pct(v, 0, True) if pd.notna(v) else "—",
                                 "rend_1m": lambda v: pct(v, 1, True), "poids": lambda v: pct(v)})
                .map(c_score, subset=["score", "mom_resid"]).map(signe_couleur, subset=["mom_12_7", "mom_12_1", "tendance_10m"]).map(c_inv, subset=["rend_1m"]),
                hide_index=True,
                column_config={"nom": "Ligne", "score": "Score", "mom_resid": "Momentum résiduel", "mom_12_7": "Momentum 12-7", "mom_12_1": "Momentum 12-1",
                               "id_fip": "Régularité (FIP)", "haut52": "% du plus haut 52 s.", "tendance_10m": "Cours / moyenne 10 mois",
                               "rend_1m": "Dernier mois", "poids": "Poids"})
        inv = sg[sg["rend_1m"] > 0.15]
        if len(inv):
            st.warning("Inversion à 1 mois : " + ", ".join(inv["nom"]) + " ont pris plus de 15 % le mois dernier. Historiquement, ces titres ont tendance "
                       "à reculer le mois suivant : ce n'est pas le moment de les renforcer.")
        sous = sg[sg["tendance_10m"] < 0]
        if len(sous):
            st.info("Sous leur moyenne sur 10 mois (Faber, 2007) : " + ", ".join(sous["nom"]) + ". La tendance de fond est baissière : à surveiller au rééquilibrage.")
        with st.expander("Comment lire chaque colonne"):
            st.markdown(
                "- **Momentum résiduel** (Blitz, Huij et Martens, 2011) : la tendance propre à l'entreprise, une fois retiré l'effet du marché (régression sur 36 mois). "
                "Presque aussi rentable que le momentum classique, avec environ deux fois moins de risque et beaucoup moins de krachs. Au-dessus de +1 : tendance propre forte.\n"
                "- **Momentum 12-7** (Novy-Marx, 2012) : la performance d'il y a 7 à 12 mois, qui prédit mieux que celle des 6 derniers mois.\n"
                "- **Momentum 12-1** : la mesure classique, sans le dernier mois.\n"
                "- **Régularité (FIP)** (Da, Gurun et Warachka, 2014) : négatif = la hausse s'est faite par petites touches régulières, ce qui persiste davantage "
                "qu'une hausse faite de quelques bonds, qui attire l'attention et se corrige.\n"
                "- **% du plus haut 52 semaines** (George et Hwang, 2004) : près de 100 %, les investisseurs hésitent à acheter au sommet, ce qui retarde la hausse suivante.\n"
                "- **Cours / moyenne 10 mois** (Faber, 2007 ; Moskowitz, Ooi et Pedersen, 2012) : au-dessus, la tendance de fond est haussière.\n"
                "- **Dernier mois** : en orange au-delà de +15 %, risque d'inversion à court terme (Jegadeesh, 1990 ; Lehmann, 1990).")

        pu_k, _, _ = prix_univers(True)
        wml_k = wml_cache(pu_k)
        if len(wml_k) > 60:
            kr_k = A.risque_krach_momentum(adj_eur[BENCH_PRINCIPAL], wml_k)
            st.markdown("### Faut-il se fier au momentum en ce moment ?")
            c = st.columns(3)
            ech = kr_k["echelle_bsc"]
            kpi(c[0], "Poids à donner au momentum", "—" if pd.isna(ech) else f"{ech:.1f}×".replace(".", ","),
                "1 = normal · sous 1 = moins de poids (Barroso et Santa-Clara, 2015)", delta_color="off")
            kpi(c[1], "Volatilité du facteur momentum", pct(kr_k["vol_wml"], 0), "6 derniers mois · cible 12 %", delta_color="off")
            kpi(c[2], "Marché sur 2 ans", pct(kr_k["marche_2ans"], 0, True),
                "baissier : terrain des krachs du momentum" if kr_k["marche_baissier"] else "pas baissier (Daniel et Moskowitz, 2016)", delta_color="off")
            if kr_k["alerte"]:
                st.error("Risque de krach du momentum : marché baissier sur 2 ans et volatilité forte. Ne t'appuie pas sur le momentum pour tes prochains choix.")
            elif pd.notna(ech) and ech < 0.8:
                st.warning("Le momentum est agité en ce moment : donne-lui moins de poids que d'habitude. Ne vends pas une ligne uniquement pour un score de "
                           "momentum faible, et ne cours pas après les titres qui viennent de flamber ; appuie-toi davantage sur la qualité, la valorisation et la diversification.")
            else:
                st.success("Contexte normal : tu peux utiliser le momentum comme d'habitude dans tes choix.")
        with st.expander("Pour aller plus loin : le facteur momentum européen en détail"):
            st.caption("Chaque mois, on achète le tiers de ta liste de départ au meilleur momentum et on vend le tiers au plus faible. Ce portefeuille "
                       "« gagnants − perdants » montre si le momentum fonctionne en ce moment sur ton univers.")
            pu, bu, _ = prix_univers(True)
            wml = wml_cache(pu)
            if len(wml) > 60:
                cumul = (1 + wml).cumprod() - 1
                vol6 = wml.rolling(126).std() * np.sqrt(252)
                g, d = st.columns(2)
                with g:
                    fig = go.Figure(go.Scatter(x=cumul.index, y=cumul, line=dict(width=2), name="Gagnants − perdants"))
                    fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
                    fig.update_layout(height=280, yaxis_tickformat="+.0%", margin=dict(t=30, l=0, r=0, b=0), title=dict(text="Rendement cumulé du facteur", font_size=13))
                    graphe(fig, key="wml")
                with d:
                    fig = go.Figure(go.Scatter(x=vol6.index, y=vol6, line=dict(width=2, color=C_ATTENTION)))
                    fig.update_layout(height=280, yaxis_tickformat=".0%", margin=dict(t=30, l=0, r=0, b=0), title=dict(text="Volatilité du facteur sur 6 mois", font_size=13))
                    graphe(fig, key="wmlvol")
                kr = A.risque_krach_momentum(adj_eur[BENCH_PRINCIPAL], wml)
                c = st.columns(4)
                kpi(c[0], "Marché sur 2 ans", pct(kr["marche_2ans"], 0, True), "baissier : danger" if kr["marche_baissier"] else "pas baissier")
                kpi(c[1], "Vol. du marché", pct(kr["vol_marche"], 0), f"6 mois · alerte au-delà de {pct(kr['seuil_vol'], 0)}")
                kpi(c[2], "Vol. du momentum", pct(kr["vol_wml"], 0), "6 derniers mois")
                kpi(c[3], "Exposition conseillée", "—" if pd.isna(kr["echelle_bsc"]) else f"{kr['echelle_bsc']:.1f}×".replace(".", ","),
                            "Barroso et Santa-Clara (2015)", delta_color="off")
                if kr["alerte"]:
                    st.error("Risque de krach du momentum : marché en baisse sur 2 ans et volatilité forte. C'est dans ce contexte que le momentum s'effondre, "
                             "quand les perdants rebondissent violemment (Daniel et Moskowitz, 2016). Ne t'appuie pas sur le momentum pour tes prochains achats.")
                else:
                    st.success("Pas de signal de krach du momentum : le contexte est normal pour utiliser ce signal.")
                st.caption("L'exposition conseillée vise une volatilité de 12 % pour le facteur : au-dessus de 1, le momentum est calme et fiable ; "
                           "en dessous, il faut lui donner moins de poids dans tes choix.")
            else:
                st.info("Pas assez d'historique pour construire le facteur.")
            st.caption("Limites honnêtes : l'avantage des anomalies a baissé d'un tiers à la moitié après leur publication (McLean et Pontiff, 2016), "
                       "et sur 25 lignes et 5 mois, le bruit domine. Le momentum sert à choisir et à contrôler, pas à promettre une performance.")


    st.markdown("### Le signal fonctionne-t-il sur ton univers ?")
    intro("Chaque mois, on mesure la corrélation de rang entre un signal et le rendement du mois suivant sur les 66 titres de ta liste : c'est le "
          "coefficient d'information (IC). Un IC moyen de 0,05 est déjà un bon signal (Grinold et Kahn, 2000). C'est le seul moyen de vérifier "
          "un signal sur tes propres titres plutôt que de croire la littérature sur parole.")
    pu_i, bu_i, _ = prix_univers(True)
    ic = ic_cache(pu_i)
    if ic.empty:
        st.info("Pas assez d'historique.")
    else:
        moy, sd, n_ = ic.mean(), ic.std(), ic.count()
        res_ic = pd.DataFrame({"ic_moyen": moy, "t": moy / (sd / np.sqrt(n_)), "positif": (ic > 0).mean(), "mois": n_})
        g, d = st.columns([2, 3])
        with g:
            tableau(res_ic.rename_axis("signal").reset_index().style.format({"ic_moyen": lambda v: f"{v:+.3f}".replace(".", ","), "t": lambda v: f"{v:+.1f}".replace(".", ","),
                                         "positif": lambda v: pct(v, 0), "mois": "{:.0f}"}).map(signe_couleur, subset=["ic_moyen"]),
                    column_config={"signal": "Signal", "ic_moyen": "IC moyen", "t": "t", "positif": "Mois où IC > 0", "mois": "Mois"})
            st.caption("Le signal « Dernier mois » est inversé : un IC positif confirme l'effet d'inversion à un mois. Un t au-delà de 2 rend le signal crédible.")
        with d:
            fig = go.Figure()
            for i, c in enumerate(ic.columns):
                fig.add_trace(go.Scatter(x=ic.index, y=ic[c].cumsum(), name=c, line=dict(width=2, color=PALETTE[i])))
            fig.add_hline(y=0, line_color="#9aa4b2", line_width=1)
            fig.update_layout(height=300, margin=dict(t=30, l=0, r=0, b=0), legend=dict(orientation="h", y=-0.15),
                              title=dict(text="IC cumulé : une courbe qui monte = un signal qui marche", font_size=13))
            graphe(fig, key="ic")

    st.markdown("### Le score de tes lignes semaine après semaine")
    intro("La carte montre l'évolution du score de momentum sur 26 semaines. Une ligne qui passe du vert au rouge perd sa tendance : "
          "c'est souvent visible plusieurs semaines avant que le momentum 12-1 ne franchisse ta limite de sortie.")
    lignes_h = [t for t in pf.index if t in adj_eur.columns]
    shs = scores_hist_cache(adj_eur[lignes_h], adj_eur[BENCH_PRINCIPAL])
    if not shs.empty:
        shs = shs.loc[shs.iloc[:, -1].sort_values(ascending=False).index]
        fig = px.imshow(shs.values, x=[f"{d:%d/%m}" for d in shs.columns], y=[pf["nom"].get(t, t) for t in shs.index],
                        color_continuous_scale="RdYlGn", zmin=-1.5, zmax=1.5, aspect="auto")
        fig.update_layout(height=40 + 24 * len(shs), margin=dict(t=10, l=0, r=0, b=0), coloraxis_colorbar=dict(title="score"))
        graphe(fig, key="scores_hist")

# ================================================================== RÉGIME
@st.cache_data(ttl=3600, show_spinner="Calcul du régime de marché…")
def regime_cache(prix, secteurs):
    r = A.rendements(prix)
    sect = pd.DataFrame({s: r[[t for t in secteurs.index[secteurs == s] if t in r.columns]].mean(axis=1) for s in secteurs.unique()}).dropna(how="any")
    sect = sect.loc[:, sect.std() > 0]
    return sect, A.turbulence(sect), A.ratio_absorption(sect)


def page_regime():
    st.subheader("Régime de marché")
    st.caption("Deux mesures de stress utilisées par les gérants institutionnels, calculées sur les secteurs de ta liste de départ, "
               "et une mesure honnête de ta diversification.")
    pu, bu, uu = prix_univers(True)
    sect, turb, absr = regime_cache(pu, uu["secteur"].reindex(pu.columns).fillna("?"))
    if len(turb) > 50:
        st.markdown("### Indice de turbulence (Kritzman et Li, 2010)")
        st.caption("Mesure à quel point les mouvements du jour entre secteurs sont inhabituels au vu de leurs corrélations de l'année passée. "
                   "Une valeur élevée signale des journées anormales, typiques des débuts de crise.")
        t20 = turb.rolling(20).mean()
        p75, p95 = turb.quantile(0.75), turb.quantile(0.95)
        actuel = float(t20.iloc[-1])
        rang = float((turb <= actuel).mean())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=turb.index, y=turb, line=dict(width=1, color="#c7c7c7"), name="Quotidien"))
        fig.add_trace(go.Scatter(x=t20.index, y=t20, line=dict(width=2), name="Moyenne 20 jours"))
        fig.add_hline(y=p75, line_dash="dot", line_color=C_ATTENTION, annotation_text="75e centile")
        fig.add_hline(y=p95, line_dash="dot", line_color=C_MAUVAIS, annotation_text="95e centile")
        fig.update_layout(height=300, margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=1.12), yaxis_type="log")
        graphe(fig, key="turb")
        c = st.columns(3)
        kpi(c[0], "Turbulence actuelle (20 jours)", f"{actuel:.2f}".replace(".", ","), f"plus haut que {pct(rang, 0)} des jours")
        etat = "Calme" if actuel < p75 else ("Agité" if actuel < p95 else "Turbulent")
        kpi(c[1], "État", etat)
        kpi(c[2], "Seuil turbulent (95e centile)", f"{p95:.2f}".replace(".", ","))
        st.caption("En régime turbulent, les corrélations montent et la diversification fond : c'est le moment de vérifier tes stress tests et ton cash.")
    if len(absr) > 20:
        st.markdown("### Ratio d'absorption (Kritzman, Li, Page et Rigobon, 2011)")
        st.caption("La part de la variance de tous les secteurs expliquée par un ou deux facteurs communs. Quand il monte, les marchés sont « soudés » : "
                   "un choc se propage partout. Les auteurs l'ont vu monter avant la plupart des grandes baisses.")
        ar15, ar_an = absr.iloc[-3:].mean(), absr.iloc[-50:]
        delta = (ar15 - ar_an.mean()) / ar_an.std() if ar_an.std() > 0 else np.nan
        fig = go.Figure(go.Scatter(x=absr.index, y=absr, line=dict(width=2, color="#7b4ea3")))
        fig.update_layout(height=260, yaxis_tickformat=".0%", margin=dict(t=10, l=0, r=0, b=0))
        graphe(fig, key="absr")
        c = st.columns(2)
        kpi(c[0], "Ratio d'absorption", pct(float(absr.iloc[-1]), 0))
        kpi(c[1], "Écart à sa moyenne de l'année", "—" if pd.isna(delta) else f"{delta:+.1f} écart-type".replace(".", ","),
                    "au-delà de +1 : fragilité" if pd.notna(delta) and delta > 1 else "normal", delta_color="off")
    st.markdown("### Combien de paris fais-tu vraiment ?")
    r1 = adj_eur.pct_change(fill_method=None).iloc[-252:]
    enb = A.nombre_effectif_paris(poids, r1)
    if enb:
        c = st.columns(3)
        kpi(c[0], "Lignes", enb["n_lignes"])
        kpi(c[1], "Paris réellement indépendants", f"{enb['enb']:.1f}".replace(".", ","))
        kpi(c[2], "Ratio de diversification", f"{enb['dr']:.2f}".replace(".", ","))
        st.caption("Le nombre de paris indépendants est le carré du ratio de diversification (Choueifaty et Coignard, 2008). Tes lignes bougeant en partie "
                   "ensemble, elles valent moins de paris que leur nombre. C'est la mesure à citer devant le jury plutôt que « 25 lignes ».")

# ================================================================== RÈGLES
def page_regles():
    st.subheader("Règles et conformité")
    st.markdown("### Chaque ligne passée au crible de tes règles")
    st.caption("✓ respectée · ✗ non respectée · – sans objet (par exemple la dette pour une banque). Les règles d'entrée servent à acheter ; "
               "une ligne déjà détenue qui en perd une est à réexaminer au prochain rééquilibrage, pas à vendre dans la minute.")
    rl = regles_lignes.sort_values("ecarts", ascending=False).copy()

    def c_sym(v):
        return {"✓": f"color:{C_BON};font-weight:600", "✗": f"color:{C_MAUVAIS};font-weight:700"}.get(v, f"color:{C_NEUTRE}")

    cols_r = ["momentum", "loterie", "roe", "dette", "valorisation", "correlation", "taille", "ligne_risquee"]
    tableau(rl[["nom"] + cols_r + ["ecarts"]].style.map(c_sym, subset=cols_r), hide_index=True,
            column_config={"nom": "Ligne", "momentum": "Momentum > 0", "loterie": "Hausse max ≤ 8 %", "roe": "ROE ≥ 10 %", "dette": "Dette ≤ 3",
                           "valorisation": "Valorisation", "correlation": "Corrélation < 0,7", "taille": "Taille", "ligne_risquee": "Règle lignes risquées",
                           "ecarts": "Écarts"})
    det = rl[rl["detail_risquee"] != ""]
    if len(det):
        st.markdown("**Lignes risquées qui ne remplissent pas toutes les conditions**")
        for _, r in det.iterrows():
            st.markdown(f"- **{r['nom']}** : {r['detail_risquee']}.")
    with st.expander("Moat retenu pour chaque ligne (à valider par toi)"):
        mo = pf[["nom", "moat"]].copy()
        mo["moat"] = mo["moat"].fillna("").replace("", "aucun moat clair")
        tableau(mo, hide_index=True, column_config={"nom": "Ligne", "moat": "Source de l'avantage durable"})
        st.caption("Les cinq sources de moat de Morningstar : avantage de coût, effet de réseau, coûts de changement, actifs immatériels, taille efficace. "
                   "Modifie la colonne « moat » de lignes.csv si tu n'es pas d'accord.")
    st.markdown("### Mes règles écrites")
    tableau(pd.DataFrame(REGLES_TEXTE, columns=["domaine", "regle", "seuil"]), hide_index=True,
            column_config={"domaine": "Domaine", "regle": "Critère", "seuil": "Seuil"})
    st.markdown("### Pourquoi le moat sur les lignes risquées")
    st.markdown("Les titres volatils et peu rentables ont des rendements futurs particulièrement mauvais (Ang et al., 2006 ; Asness, Frazzini et Pedersen, 2019), "
                "et les titres « loterie » sous-performent (Bali, Cakici et Whitelaw, 2011). Exiger un avantage durable et une rentabilité élevée là où le risque "
                "est le plus fort limite ce piège. Un narratif seul ne suffit pas : les actions « glamour » sous-performent (Lakonishok, Shleifer et Vishny, 1994).")

# ================================================================== CANDIDATS
def page_candidats():
    st.subheader("Écran des candidats")
    st.caption("Tout l'univers (environ 135 grandes valeurs européennes, hors tes lignes) passé au crible de tes règles d'entrée, avec ce que chaque titre "
               "ferait à ton portefeuille. Un bon candidat respecte les règles et fait baisser le risque ou monter la diversification.")
    pu, bu, uu = prix_univers(False)
    if bu is None:
        st.error("Indice de référence indisponible.")
    else:
        detenus = set(pf.index)
        cand = [t for t in pu.columns if t not in detenus and pu[t].dropna().shape[0] > 260]
        c1, c2, c3 = st.columns([2, 2, 1])
        mode_txt = c1.radio("Test", ["Ajout à son poids cible", "Remplacement d'une ligne"], horizontal=True)
        a_remplacer = None
        if mode_txt.startswith("Remplacement"):
            a_remplacer = c2.selectbox("Ligne remplacée", pf.sort_values("nom").index, format_func=lambda t: pf.loc[t, "nom"])
        secteurs_dispo = ["Tous"] + sorted(uu["secteur"].dropna().unique().tolist())
        sect_f = c3.selectbox("Secteur", secteurs_dispo)
        f1, f2 = st.columns(2)
        seulement_ok = f1.checkbox("Seulement les candidats sans aucun écart aux règles", value=False)
        avec_fond = f2.checkbox("Ajouter les fondamentaux Yahoo (ROE, dette, PEG)", value=False,
                                help="Une à deux minutes la première fois, puis gardés 24 h. Chiffres indicatifs : vérifie sur MarketScreener avant d'acheter.")

        tous = pu.join(bu.rename(BENCH_PRINCIPAL)) if BENCH_PRINCIPAL not in pu.columns else pu
        st_u = stats_lignes(tous, BENCH_PRINCIPAL)
        sg_u = signaux_cache(pu[[c for c in pu.columns if pu[c].dropna().shape[0] > 260]], bu)
        R = adj_eur.join(pu[[c for c in pu.columns if c not in adj_eur.columns]], how="outer").pct_change(fill_method=None).iloc[-252:]
        a_cible = {c: (POIDS_REDUIT if pd.notna(st_u.loc[c, "vol_ewma"]) and st_u.loc[c, "vol_ewma"] > LIM["vol_reduite"] else POIDS_BASE)
                   for c in cand if c in st_u.index}
        imp, vol0, dr0 = impact_candidats(poids, cand, R, "ajout" if a_remplacer is None else "remplacement", a_cible, a_remplacer)
        rp = (R[[t for t in poids.index if t in R.columns]].fillna(0) @ poids.reindex([t for t in poids.index if t in R.columns]).fillna(0))
        # poids sectoriels et par place après l'opération
        sec_now = pf.groupby(pf["secteur"].fillna("?"))["poids"].sum()
        pl_now = pf.groupby(pf["place"].fillna("?"))["poids"].sum()
        rows = []
        for c in cand:
            if c not in imp.index or c not in st_u.index:
                continue
            sct, plc = uu.loc[c, "secteur"], uu.loc[c, "place"]
            a = a_cible.get(c, POIDS_BASE)
            retire_s = retire_p = 0.0
            if a_remplacer is not None:
                a = float(pf.loc[a_remplacer, "poids"])
                retire_s = a if pf.loc[a_remplacer, "secteur"] == sct else 0.0
                retire_p = a if pf.loc[a_remplacer, "place"] == plc else 0.0
            sec_apres = sec_now.get(sct, 0.0) * (1 if a_remplacer else (1 - a)) - retire_s + a
            pl_apres = pl_now.get(plc, 0.0) * (1 if a_remplacer else (1 - a)) - retire_p + a
            corr_l = R[[t for t in pf.index if t in R.columns]].corrwith(R[c]).drop(labels=[a_remplacer] if a_remplacer else [], errors="ignore")
            rows.append({
                "ticker": c, "nom": uu.loc[c, "nom"], "secteur": sct, "place": plc,
                "score": sg_u["score"].get(c, np.nan), "momentum": st_u.loc[c, "momentum"], "rend_1m": sg_u["rend_1m"].get(c, np.nan),
                "max_jour": st_u.loc[c, "max_jour"], "vol_ewma": st_u.loc[c, "vol_ewma"], "beta": st_u.loc[c, "beta"],
                "corr_max": corr_l.max(), "corr_ptf": R[c].corr(rp),
                "d_vol": imp.loc[c, "vol_apres"] - vol0, "d_dr": imp.loc[c, "dr_apres"] - dr0,
                "secteur_apres": sec_apres, "place_apres": pl_apres, "poids_prevu": a,
            })
        cd = pd.DataFrame(rows).set_index("ticker")
        if avec_fond and len(cd):
            fo = fondamentaux_yahoo(tuple(cd.index))
            cd = cd.join(fo[["roe", "dette", "per", "croissance", "peg"]], how="left")
        else:
            for k in ["roe", "dette", "per", "croissance", "peg"]:
                cd[k] = np.nan

        def ecarts(r):
            e = []
            if not (pd.notna(r["momentum"]) and r["momentum"] > 0):
                e.append("momentum")
            if pd.notna(r["max_jour"]) and r["max_jour"] > 0.08:
                e.append("loterie")
            if pd.notna(r["corr_max"]) and r["corr_max"] >= 0.7:
                e.append("corrélation")
            if r["secteur_apres"] > LIM["secteur"] + 1e-9:
                e.append("secteur plein")
            if r["place_apres"] > LIM["place"] + 1e-9:
                e.append("place pleine")
            if pd.notna(r["rend_1m"]) and r["rend_1m"] > 0.15:
                e.append("inversion à 1 mois")
            if pd.notna(r["roe"]) and r["roe"] < 0.10:
                e.append("ROE")
            if pd.notna(r["dette"]) and r["dette"] > 3 and r["secteur"] not in ("Finance", "Utilities"):
                e.append("dette")
            if pd.notna(r["peg"]) and r["peg"] > 2:
                e.append("PEG")
            if pd.notna(r["vol_ewma"]) and r["vol_ewma"] > LIM["vol_reduite"]:
                e.append("ligne risquée : moat à vérifier")
            return ", ".join(e)

        if len(cd):
            cd["ecarts"] = cd.apply(ecarts, axis=1)
            cd["n_ecarts"] = cd["ecarts"].map(lambda x: 0 if not x else len(x.split(", ")))
            if sect_f != "Tous":
                cd = cd[cd["secteur"] == sect_f]
            if seulement_ok:
                cd = cd[cd["n_ecarts"] == 0]
            cd = cd.sort_values(["n_ecarts", "d_dr", "score"], ascending=[True, False, False])
            st.caption(f"Portefeuille actuel : volatilité {pct(vol0)}, ratio de diversification {dr0:.2f}. ".replace(".", ",", 1) +
                       ("Chaque candidat entre à son poids cible, le reste du portefeuille est réduit d'autant." if a_remplacer is None
                        else f"Chaque candidat prend la place de {pf.loc[a_remplacer, 'nom']} ({pct(pf.loc[a_remplacer, 'poids'])})."))
            vue = cd.head(40).copy()
            vue["ecarts"] = vue["ecarts"].replace("", "aucun")

            def c_dvol(v):
                return "" if pd.isna(v) else (f"color:{C_BON};font-weight:600" if v < -0.001 else (f"color:{C_MAUVAIS}" if v > 0.001 else ""))

            def c_ddr(v):
                return "" if pd.isna(v) else (f"color:{C_BON};font-weight:600" if v > 0.01 else (f"color:{C_MAUVAIS}" if v < -0.01 else ""))

            def c_ec(v):
                return f"color:{C_BON};font-weight:600" if v == "aucun" else f"color:{C_ATTENTION}"

            n2 = lambda v: "—" if pd.isna(v) else f"{v:+.2f}".replace(".", ",")
            tableau(vue[["nom", "secteur", "place", "ecarts", "d_dr", "d_vol", "corr_ptf", "corr_max", "score", "momentum", "rend_1m", "vol_ewma",
                         "roe", "dette", "peg", "poids_prevu"]].style.format({
                "d_dr": n2, "d_vol": lambda v: pct(v, 2, True), "corr_ptf": lambda v: "—" if pd.isna(v) else f"{v:.2f}".replace(".", ","),
                "corr_max": lambda v: "—" if pd.isna(v) else f"{v:.2f}".replace(".", ","), "score": n2, "momentum": lambda v: pct(v, 0, True),
                "rend_1m": lambda v: pct(v, 1, True), "vol_ewma": lambda v: pct(v, 0), "roe": lambda v: pct(v, 0), "dette": lambda v: "—" if pd.isna(v) else f"{v:.1f}".replace(".", ","),
                "peg": lambda v: "—" if pd.isna(v) else f"{v:.2f}".replace(".", ","), "poids_prevu": lambda v: pct(v)})
                .map(c_dvol, subset=["d_vol"]).map(c_ddr, subset=["d_dr"]).map(c_ec, subset=["ecarts"]).map(signe_couleur, subset=["momentum"]),
                hide_index=True,
                column_config={"nom": "Candidat", "secteur": "Secteur", "place": "Place", "ecarts": "Écarts aux règles", "d_dr": "Δ diversification",
                               "d_vol": "Δ volatilité", "corr_ptf": "Corr. avec ton portefeuille", "corr_max": "Corr. max avec une ligne", "score": "Score momentum",
                               "momentum": "Momentum 12-1", "rend_1m": "Dernier mois", "vol_ewma": "Vol EWMA", "roe": "ROE", "dette": "Dette nette / Ebitda",
                               "peg": "PEG", "poids_prevu": "Poids prévu"})
            st.caption(f"{len(cd)} candidats affichés sur {len(rows)}, les 40 premiers ci-dessus. Classement : d'abord le moins d'écarts aux règles, "
                       "puis le plus fort gain de diversification, puis le meilleur momentum.")
            with st.expander("Comment choisir"):
                st.markdown(
                    "1. **Aucun écart aux règles** d'abord : sinon tu prends une exception à justifier.\n"
                    "2. **Δ diversification positif et Δ volatilité négatif** : le titre apporte quelque chose que tu n'as pas déjà. Une corrélation faible avec "
                    "ton portefeuille va dans le même sens.\n"
                    "3. **Score de momentum** : à diversification égale, prends la tendance la plus solide.\n"
                    "4. Puis **vérifie à la main** : fondamentaux sur MarketScreener, prochains résultats, et écris la thèse, la menace et la condition de sortie.\n"
                    "5. Teste le finaliste dans le **Simulateur** avant de passer l'ordre.\n\n"
                    "Le moat ne se mesure pas automatiquement : pour une ligne risquée (vol EWMA au-dessus de 40 %), c'est à toi de l'identifier.")

# ================================================================== RÉÉQUILIBRAGE
def page_reequilibrage():
    st.subheader("Rééquilibrage")
    st.caption("Calcule les ordres pour revenir aux poids cibles, après les sorties et les entrées que tu choisis, avec une bande de tolérance pour ne pas "
               "trader pour quelques euros. Les prix sont ceux de Yahoo : les montants réels sur NextWise différeront un peu.")
    sorties_regle = [t for t, r in pf.iterrows() if pd.notna(r.get("momentum")) and r["momentum"] < LIM["momentum"]]
    c1, c2 = st.columns(2)
    a_vendre = c1.multiselect("Lignes à vendre entièrement", list(pf.index), default=sorties_regle, format_func=lambda t: pf.loc[t, "nom"], placeholder="Aucune",
                              help="Par défaut : les lignes dont le momentum 12-1 est sous −20 % (ta règle de sortie).")
    univ_all = charger_univers()
    choix_ach = [t for t in univ_all.index if t not in pf.index]
    a_acheter = c2.multiselect("Nouvelles lignes à acheter", choix_ach, format_func=lambda t: f"{univ_all.loc[t, 'nom']} ({t})", placeholder="Choisis un ou plusieurs titres")
    libres = c2.text_input("Autres tickers Yahoo, hors univers (séparés par des virgules)", key="tick_libres", placeholder="ex. SOI.PA, NEXI.MI")
    libres = [x.strip().upper() for x in libres.split(",") if x.strip() and x.strip().upper() not in pf.index]
    c3, c4 = st.columns(2)
    bande = c3.slider("Bande de tolérance (points de poids)", 0.0, 2.0, 0.5, 0.1,
                      help="On ne corrige une ligne que si son poids s'écarte de la cible de plus de cette valeur.") / 100
    cash_cible = c4.slider("Cash visé (% du total)", 0.0, 10.0, 2.0, 0.5) / 100
    # cibles
    cibles = {}
    for t, r in pf.iterrows():
        if t in a_vendre:
            cibles[t] = 0.0
            continue
        base_c = r["poids_cible"] if pd.notna(r.get("poids_cible")) else POIDS_BASE
        if pd.notna(r.get("vol_ewma")) and r["vol_ewma"] > LIM["vol_reduite"]:
            base_c = min(base_c, POIDS_REDUIT)
        cibles[t] = base_c
    prix_new, nom_new, sect_new, place_new = {}, {}, {}, {}
    if libres:
        dev_l = {t: devise_de(t) for t in libres}
        pl_ = sorted({("EURGBP=X" if d == "GBp" else f"EUR{d}=X") for d in dev_l.values() if d != "EUR"})
        cl_l, _ = historique(tuple(sorted(set(libres) | set(pl_))))
        cl_l = en_euros(cl_l, dev_l) if not cl_l.empty else cl_l
        for t in libres:
            if t in cl_l.columns and cl_l[t].dropna().shape[0]:
                prix_new[t] = float(cl_l[t].dropna().iloc[-1])
                nom_new[t], sect_new[t], place_new[t] = t, "Non renseigné", "Non renseignée"
                cibles[t] = POIDS_BASE
            else:
                st.warning(f"Aucun cours Yahoo pour {t} : vérifie le ticker.")
    if a_acheter:
        pu2, bu2, uu2 = prix_univers(False)
        st_n = stats_lignes(pu2[[t for t in a_acheter if t in pu2.columns]].join(bu2.rename(BENCH_PRINCIPAL)), BENCH_PRINCIPAL) if bu2 is not None else pd.DataFrame()
        for t in a_acheter:
            if t in pu2.columns and pu2[t].dropna().shape[0]:
                prix_new[t] = float(pu2[t].dropna().iloc[-1])
                nom_new[t], sect_new[t], place_new[t] = uu2.loc[t, "nom"], uu2.loc[t, "secteur"], uu2.loc[t, "place"]
                v = st_n["vol_ewma"].get(t, np.nan) if len(st_n) else np.nan
                cibles[t] = POIDS_REDUIT if pd.notna(v) and v > LIM["vol_reduite"] else POIDS_BASE
    somme = sum(cibles.values())
    if somme > 1 - cash_cible:
        k = (1 - cash_cible) / somme
        cibles = {t: x * k for t, x in cibles.items()}
        st.caption(f"Les cibles dépassaient {pct(1 - cash_cible, 0)} investi : elles ont été réduites proportionnellement (× {k:.3f}).".replace(".", ",", 1))
    ordres = []
    for t, cible in cibles.items():
        if t in pf.index:
            r = pf.loc[t]
            cours, qte, poids_av = r["cours"], r["quantite"], r["poids"]
            nom_ = r["nom"]
        else:
            cours, qte, poids_av, nom_ = prix_new.get(t, np.nan), 0.0, 0.0, nom_new.get(t, t)
        if pd.isna(cours) or cours <= 0:
            continue
        if cible == 0 and qte > 0:
            dq = -qte
        elif abs(cible - poids_av) <= bande and qte > 0:
            dq = 0
        else:
            dq = int(np.floor((cible * total - qte * cours) / cours)) if cible > poids_av else -int(np.floor((qte * cours - cible * total) / cours))
        ordres.append({"ticker": t, "nom": nom_, "poids_avant": poids_av, "cible": cible, "quantite_avant": qte, "ordre": dq,
                       "sens": "Achat" if dq > 0 else ("Vente" if dq < 0 else "—"), "cours": cours, "montant": dq * cours,
                       "quantite_apres": qte + dq})
    od = pd.DataFrame(ordres).set_index("ticker")
    cash_apres = cash - od["montant"].sum()
    od["poids_apres"] = od["quantite_apres"] * od["cours"] / total
    m = st.columns(4)
    kpi(m[0], "Ordres à passer", int((od["ordre"] != 0).sum()))
    kpi(m[1], "Montant échangé", eur(od["montant"].abs().sum()), f"{pct(od['montant'].abs().sum() / total)} du portefeuille")
    kpi(m[2], "Cash après", eur(cash_apres), f"{pct(cash_apres / total)} du total")
    if cash_apres < 0:
        kpi(m[3], "Attention", "cash négatif")
        st.error("Les achats dépassent le cash disponible : réduis les cibles ou vends davantage.")
    elif cash_apres / total > cash_cible + 0.02:
        st.info(f"Il resterait {pct(cash_apres / total)} de cash pour une cible de {pct(cash_cible, 0)} : choisis des remplaçants dans « Nouvelles lignes à acheter » "
                "(l'écran des candidats t'aide à les trouver).")
    passe = od[od["ordre"] != 0].copy()
    if len(passe):
        st.markdown("### Ordres proposés")
        tableau(passe[["nom", "sens", "ordre", "cours", "montant", "poids_avant", "poids_apres", "cible"]].style.format(
            {"ordre": lambda v: f"{int(v):+d}", "cours": lambda v: eur(v, 2), "montant": lambda v: eur(v), "poids_avant": lambda v: pct(v),
             "poids_apres": lambda v: pct(v), "cible": lambda v: pct(v)}).map(lambda v: f"color:{C_BON};font-weight:600" if v == "Achat" else (f"color:{C_MAUVAIS};font-weight:600" if v == "Vente" else ""), subset=["sens"]),
            hide_index=True, column_config={"nom": "Ligne", "sens": "Sens", "ordre": "Quantité", "cours": "Cours estimé", "montant": "Montant",
                                            "poids_avant": "Poids avant", "poids_apres": "Poids après", "cible": "Cible"})
    else:
        st.success("Aucun ordre nécessaire : toutes les lignes sont dans la bande de tolérance.")
    # contraintes et risque après
    apres = od[od["quantite_apres"] > 0].copy()
    apres["poids"] = apres["poids_apres"]
    apres["secteur"] = [pf["secteur"].get(t) if t in pf.index else sect_new.get(t) for t in apres.index]
    apres["place"] = [pf["place"].get(t) if t in pf.index else place_new.get(t) for t in apres.index]
    st.markdown("### Après les ordres")
    tableau(contraintes(apres).style.format({"niveau": lambda v: pct(v), "limite": lambda v: "≤ " + pct(v, 0)}).map(couleur_etat, subset=["etat"]),
            hide_index=True, column_config={"regle": "Règle", "niveau": "Niveau", "limite": "Limite", "etat": "État"})
    adj_r = adj_eur
    if a_acheter:
        pu2, _, _ = prix_univers(False)
        adj_r = adj_eur.join(pu2[[t for t in a_acheter if t in pu2.columns and t not in adj_eur.columns]], how="left")
    if libres:
        _, ad_l = historique(tuple(sorted(set(libres) | {f"EUR{devise_de(t)}=X" for t in libres if devise_de(t) not in ("EUR", "GBp")})))
        if not ad_l.empty:
            ad_l = en_euros(ad_l, {t: devise_de(t) for t in libres})
            adj_r = adj_r.join(ad_l[[t for t in libres if t in ad_l.columns and t not in adj_r.columns]], how="left")
    r_ap = risque_portefeuille(apres["poids"], adj_r, BENCH_PRINCIPAL)
    if r_ap and risque:
        c = st.columns(3)
        kpi(c[0], "Volatilité prévue", pct(r_ap["vol"]), pct(r_ap["vol"] - risque["vol"], 2, True), delta_color="inverse")
        kpi(c[1], "Ratio de diversification", f"{r_ap['dr']:.2f}".replace(".", ","), f"{r_ap['dr'] - risque['dr']:+.2f}".replace(".", ","))
        kpi(c[2], "Bêta", f"{r_ap['beta']:.2f}".replace(".", ","), f"{r_ap['beta'] - risque['beta']:+.2f}".replace(".", ","))
    if len(passe):
        st.markdown("### Lignes à copier dans le journal")
        st.caption("Une fois les ordres exécutés sur NextWise, remplace les prix estimés par les prix réels (montant ÷ quantité), puis colle ces lignes à la fin de transactions.csv.")
        jour = dt.datetime.now(TZ).strftime("%Y-%m-%d")
        texte = "\n".join(f"{jour},{t},{'Achat' if r['ordre'] > 0 else 'Vente'},{abs(int(r['ordre']))},{r['cours']:.4f},,prix estimé à corriger" for t, r in passe.iterrows())
        st.code(texte, language=None)
    st.caption("La bande de tolérance limite les allers-retours inutiles : corriger un écart de quelques dixièmes de point coûte plus en frais et en "
               "bruit qu'il ne rapporte (Masters, 2003 ; Leland, 1999).")


    st.markdown("### Test de la page blanche")
    intro("Pour chaque ligne : si tu ne l'avais pas, tes règles te la feraient-elles acheter aujourd'hui ? Une ligne qui échoue n'est pas forcément à vendre, "
          "mais tu dois pouvoir justifier par écrit pourquoi tu la gardes. C'est la parade la plus simple à l'effet de disposition.")
    pb = regles_lignes[regles_lignes["ecarts"] > 0]
    if pb.empty:
        st.success("Toutes tes lignes passent le test : tes règles te les feraient toutes acheter aujourd'hui.")
    else:
        noms_r = {"momentum": "momentum négatif", "loterie": "hausse de plus de 8 % en un jour", "roe": "ROE sous 10 %", "dette": "dette au-dessus de 3",
                  "valorisation": "valorisation hors critères", "correlation": "corrélation de 0,7 ou plus", "taille": "poids trop élevé",
                  "ligne_risquee": "règle des lignes risquées"}
        for t, r in pb.sort_values("ecarts", ascending=False).iterrows():
            raisons = [v for k_, v in noms_r.items() if r[k_] == "✗"]
            verdict = pf["verdict"].get(t)
            st.markdown(f"- **{r['nom']}** : {', '.join(raisons)}." + (f" Décision déjà écrite : *{verdict}*." if isinstance(verdict, str) and verdict else ""))

# ================================================================== SIMULATEUR
def page_simulateur():
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
    univ_s = charger_univers()
    for col_ in ["nom", "secteur", "place"]:
        edit[col_] = [v if isinstance(v, str) and v.strip() else (univ_s[col_].get(t) if t in univ_s.index else (t if col_ == "nom" else None))
                      for t, v in zip(edit["ticker"], edit[col_])]
    edit["poids"] = edit["poids"].fillna(0) / 100
    sans_info = [t for t, s_, p_ in zip(edit["ticker"], edit["secteur"], edit["place"]) if not s_ or not p_]
    if sans_info:
        st.caption("Secteur ou place inconnus pour " + ", ".join(sans_info) + " : complète-les dans le tableau pour que les contraintes soient justes.")
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
            st.markdown("### Effet de l'ordre sur le risque")
            c = st.columns(5)
            kpi(c[0], "Volatilité", pct(apres["vol"]), pct(apres["vol"] - risque["vol"], 2, True), delta_color="inverse")
            kpi(c[1], "Ratio de diversification", f"{apres['dr']:.2f}".replace(".", ","), f"{apres['dr'] - risque['dr']:+.2f}".replace(".", ","))
            kpi(c[2], "Bêta", f"{apres['beta']:.2f}".replace(".", ","), f"{apres['beta'] - risque['beta']:+.2f}".replace(".", ","))
            kpi(c[3], "VaR 95 % sur un jour", pct(apres["var95"], 2), pct(apres["var95"] - risque["var95"], 2, True), delta_color="inverse")
            kpi(c[4], "Cash", pct(1 - w_sim.sum()), pct((1 - w_sim.sum()) - cash / total, 1, True))
            st.markdown("### Contraintes après l'ordre")
            cs = contraintes(edit[edit["poids"] > 0].set_index("ticker"))
            tableau(cs.style.format({"niveau": lambda v: pct(v), "limite": lambda v: "≤ " + pct(v, 0)}).map(couleur_etat, subset=["etat"]),
                         hide_index=True, column_config={"regle": "Règle après l'ordre", "niveau": "Niveau", "limite": "Limite", "etat": "État"})
            st.markdown("### Crises rejouées, avant et après")
            s_av, s_ap = stress(poids, adj_sim, BENCH_PRINCIPAL), stress(w_sim, adj_sim, BENCH_PRINCIPAL)
            if not s_av.empty and not s_ap.empty:
                comp = s_av[["scenario", "portefeuille"]].merge(s_ap[["scenario", "portefeuille"]], on="scenario", suffixes=("_avant", "_apres"))
                comp["ecart"] = comp["portefeuille_apres"] - comp["portefeuille_avant"]
                tableau(comp.style.format({c: (lambda v: pct(v, 1, True)) for c in ["portefeuille_avant", "portefeuille_apres", "ecart"]}).map(signe_couleur, subset=["ecart"]),
                             hide_index=True, column_config={"scenario": "Crise rejouée", "portefeuille_avant": "Avant", "portefeuille_apres": "Après", "ecart": "Écart"})

# ================================================================== STRESS
def page_stress():
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
        graphe(fig, key="stress")
        tableau(st_df.style.format({"portefeuille": lambda v: pct(v, 1, True), "reference": lambda v: pct(v, 1, True), "couverture": lambda v: pct(v, 0)}),
                     hide_index=True, column_config={"scenario": "Crise", "portefeuille": "Portefeuille", "reference": "Euronext 100", "couverture": "Capital avec historique"})
    st.subheader("Choc de marché instantané")
    choc = st.slider("Baisse de l'Euronext 100", -40, 0, -10, format="%d %%")
    if risque and pd.notna(risque["beta"]):
        perte = risque["beta"] * choc / 100
        kpi(st, "Perte estimée du portefeuille", pct(perte, 1, True), eur(perte * total))
        st.caption("Estimation par le bêta : elle suppose que les corrélations restent normales, ce qui sous-estime la perte dans une vraie crise.")


    st.markdown("### Scénarios à curseurs")
    intro("L'impact estimé d'un choc combiné, à partir de la sensibilité de chacune de tes lignes au marché, au pétrole, au dollar et aux taux "
          "(régression sur deux ans de cours quotidiens).")
    mac = series_macro()
    r_m = adj_eur[BENCH_PRINCIPAL].pct_change(fill_method=None)
    morceaux = [r_m.rename("Marché (Euronext 100)")] + ([mac.pct_change(fill_method=None)] if not mac.empty else [])
    fac_s = pd.concat(morceaux, axis=1).iloc[-504:]
    R_l = adj_eur[[t for t in poids.index if t in adj_eur.columns]].pct_change(fill_method=None).reindex(fac_s.index)
    sens = A.sensibilites(R_l, fac_s.dropna())
    if sens.empty:
        st.info("Données de marché insuffisantes pour estimer les sensibilités.")
    else:
        cols_s = list(sens.columns)
        chocs = {}
        cc = st.columns(len(cols_s))
        defs = {"Marché (Euronext 100)": (-40, 20, -10), "Pétrole (Brent)": (-50, 80, 0), "Euro / dollar": (-15, 15, 0), "Obligations d'État € (prix)": (-12, 6, 0)}
        aides = {"Euro / dollar": "Positif : l'euro monte face au dollar.", "Obligations d'État € (prix)": "Environ −7 % pour une hausse de 1 point des taux longs."}
        for i, ccol in enumerate(cols_s):
            lo, hi, dv = defs.get(ccol, (-30, 30, 0))
            chocs[ccol] = cc[i].slider(ccol, lo, hi, dv, format="%d %%", key=f"choc_{i}", help=aides.get(ccol)) / 100
        imp = sens[cols_s].mul(pd.Series(chocs)).sum(axis=1)
        w_ = poids.reindex(imp.index).fillna(0)
        tot_imp = float((w_ * imp).sum())
        c = st.columns(2)
        kpi(c[0], "Impact estimé sur le portefeuille", pct(tot_imp, 1, True), eur(tot_imp * total))
        kpi(c[1], "Ligne la plus touchée", pf["nom"].get(imp.idxmin(), imp.idxmin()), pct(imp.min(), 1, True))
        contrib = (w_ * imp).sort_values()
        fig = go.Figure(go.Bar(x=contrib.values, y=[pf["nom"].get(t, t) for t in contrib.index], orientation="h",
                               marker_color=[C_BON if v >= 0 else C_MAUVAIS for v in contrib.values],
                               text=[pct(v, 2, True) for v in contrib.values], textposition="outside"))
        fig.update_layout(height=max(320, 22 * len(contrib)), xaxis_tickformat="+.1%", margin=dict(t=30, l=0, r=50, b=0),
                          title=dict(text="Contribution de chaque ligne à l'impact (poids × choc)", font_size=13))
        graphe(fig, key="chocs")
        st.caption("Modèle linéaire en temps normal : il sous-estime les vraies crises, où les corrélations montent. Les sensibilités sont estimées sur deux ans "
                   "et changent avec le temps. Pour Equinor et Ryanair, le curseur pétrole montre bien leur rôle de contrepoids.")

    st.markdown("### Les 13 scénarios d'APTimum")
    intro("Les scénarios du rapport APTimum du 30/10/2026, calculés par leur modèle de risque sur les 21 lignes détenues à cette date. "
          "Ils complètent tes crises rejouées : mêmes ordres de grandeur, méthode différente.")
    try:
        ap = pd.read_csv(os.path.join(ICI, "aptimum.csv"))
    except Exception:
        ap = pd.DataFrame()
    if not ap.empty:
        ap = ap.iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ap["scenario"], x=ap["portefeuille"], name="Portefeuille", orientation="h", marker_color=PALETTE[0],
                             text=[pct(v, 1, True) for v in ap["portefeuille"]], textposition="outside"))
        fig.add_trace(go.Bar(y=ap["scenario"], x=ap["reference"], name="Référence", orientation="h", marker_color="#9aa4b2"))
        fig.update_layout(barmode="group", height=560, xaxis_tickformat=".0%", margin=dict(t=10, l=0, r=40, b=0), legend=dict(orientation="h", y=1.04))
        graphe(fig, key="aptimum")
        mieux = int((ap["portefeuille"] > ap["reference"]).sum())
        st.caption(f"Tu perds moins que la référence dans {mieux} scénarios sur {len(ap)}. Pour mettre à jour, remplace aptimum.csv par les chiffres du nouveau rapport.")

# ================================================================== CHANCE OU TALENT
@st.cache_data(ttl=3600, show_spinner="Bootstrap en cours…")
def bootstrap_cache(r_ptf, r_marche, n_boot=2000, bloc=5):
    return A.bootstrap_alpha(r_ptf, r_marche, n_boot=n_boot, longueur_bloc=bloc)


def lire_participants(fichier):
    """Lit l'export de la carte performance-risque (CSV ou Excel). Colonnes reconnues : nom, performance, risque."""
    try:
        if hasattr(fichier, "name") and str(fichier.name).lower().endswith((".xls", ".xlsx")):
            d = pd.read_excel(fichier)
        else:
            d = pd.read_csv(fichier, sep=None, engine="python")
    except Exception as e:
        return None, f"Fichier illisible : {e}"
    return d, None


def page_chance():
    st.subheader("Chance ou talent ?")
    st.caption("La page suit la démarche de ton TFE : d'abord ce que le hasard seul peut produire (Monte Carlo), "
               "puis un test sur tes vrais rendements sans supposer de loi normale (bootstrap), enfin la correction "
               "des tests multiples à l'échelle de tous les participants (taux de fausses découvertes).")
    achats = tx.loc[tx["type"] == "achat", "date"].dropna()
    aujourdhui_d = dt.datetime.now(TZ).date()
    defaut = achats.value_counts().idxmax().date() if len(achats) else aujourdhui_d
    c1, c2, c3 = st.columns([1, 1, 2])
    d_debut = c1.date_input("Début", value=min(defaut, aujourdhui_d), max_value=aujourdhui_d, format="DD/MM/YYYY")
    d_fin = c2.date_input("Fin", value=aujourdhui_d, max_value=aujourdhui_d, format="DD/MM/YYYY")
    choix_u = c3.radio("Univers de tirage", ["Ma liste de départ", "Univers large"], horizontal=True,
                       help="Ma liste : les titres que tu avais présélectionnés. Univers large : environ 135 grandes valeurs des places européennes, plus exigeant.")
    if st.button("Lancer les 10 000 portefeuilles", type="primary"):
        st.session_state["simu_cle"] = (choix_u, d_debut, d_fin)
    cle = st.session_state.get("simu_cle")

    # ------------------------------------------------------------- ÉTAPE 1
    st.markdown("### Étape 1 · Ce que le hasard seul peut produire (Monte Carlo)")
    st.caption("10 000 portefeuilles au hasard qui suivent tes règles : même nombre de lignes, poids de 4,14 % (2,5 % si la vol EWMA dépasse 40 %), "
               "au plus 20 % par secteur et 30 % par place, même part investie. Ils sont comparés à tes lignes détenues au début de la période, conservées jusqu'à la fin "
               "(Tarnavskyi, 2020 ; Burns, 2004).")
    res = None
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
            res = None
    if res:
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
        kpi(m[0], "Ton rendement", pct(rm, 2, True), f"Euronext 100 {pct(res['bench'], 2, True)}")
        kpi(m[1], "Mieux que", pct(pct_tous, 0), "de tous les portefeuilles aléatoires")
        kpi(m[2], "À risque égal, mieux que", pct(pct_risque, 0), f"{int(proches.sum())} portefeuilles à ± {pct(bande, 0)} de vol")
        kpi(m[3], "p-valeur", f"{p_val:.2f}".replace(".", ","), "le hasard fait aussi bien dans cette part des cas")
        if p_val < 0.05:
            st.success("Sur cette période, le hasard explique difficilement ton résultat. Une seule période ne prouve pas un talent : refais le test chaque mois.")
        elif p_val > 0.5:
            st.warning("Sur cette période, des portefeuilles tirés au hasard avec tes règles font aussi bien ou mieux dans la majorité des cas.")
        else:
            st.info("Résultat compatible avec le hasard : ni preuve de talent, ni preuve du contraire. C'est normal sur une période courte.")
        g, d = st.columns(2)
        with g:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ra, nbinsx=70, name="Tous", marker_color="#c7c7c7", opacity=0.8))
            fig.add_trace(go.Histogram(x=ra[proches], nbinsx=70, name="À risque égal", marker_color="#4c78a8", opacity=0.8))
            fig.add_vline(x=rm, line_color=C_MAUVAIS, line_width=2, annotation_text="toi", annotation_position="top")
            if pd.notna(res["bench"]):
                fig.add_vline(x=res["bench"], line_color=C_NEUTRE, line_dash="dot", annotation_text="Euronext 100", annotation_position="bottom right")
            fig.update_layout(barmode="overlay", height=340, xaxis_tickformat="+.1%", margin=dict(t=30, l=0, r=0, b=0),
                              legend=dict(orientation="h", y=1.12), yaxis_title="portefeuilles")
            graphe(fig, key="alea_hist")
        with d:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=va, y=ra, mode="markers", marker=dict(size=3, color="#9e9e9e", opacity=0.35), name="Aléatoires",
                                       hovertemplate="vol %{x:.1%} · rendement %{y:+.2%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[vm], y=[rm], mode="markers+text", marker=dict(size=16, color=C_MAUVAIS, symbol="star"),
                                     text=["toi"], textposition="top center", name="Ton portefeuille"))
            fig.update_layout(height=340, xaxis_tickformat=".0%", yaxis_tickformat="+.1%", xaxis_title="volatilité prévue",
                              yaxis_title="rendement", margin=dict(t=30, l=0, r=0, b=0), showlegend=False)
            graphe(fig, key="alea_nuage")
        if res.get("survie") is not None and not res["survie"].empty:
            st.markdown("**Tableau de survie, mois par mois** (la méthode de Tarnavskyi appliquée à toi)")
            st.caption("Chaque mois : la part des portefeuilles aléatoires qui battent l'indice, et ta place parmi eux. Battre l'indice un mois où 80 % des tirages le battent ne dit rien de toi.")
            sv = res["survie"].copy()
            sv["tu_bats_indice"] = sv["tu_bats_indice"].map({True: "Oui", False: "Non"})
            tableau(sv.style.format({"indice": lambda v: pct(v, 2, True), "part_alea_bat_indice": lambda v: pct(v, 0),
                                     "toi": lambda v: pct(v, 2, True), "ton_percentile": lambda v: pct(v, 0)}),
                    hide_index=True, column_config={"periode": "Mois", "indice": "Euronext 100", "part_alea_bat_indice": "Tirages qui battent l'indice",
                                                    "toi": "Toi", "tu_bats_indice": "Tu bats l'indice", "ton_percentile": "Ta place parmi les tirages"})
        notes = []
        if res["exclus_moi"]:
            notes.append("Lignes sans assez d'historique, comptées comme du cash : " + ", ".join(res["exclus_moi"]) + ".")
        if res["incomplets"]:
            notes.append(f"{res['incomplets']} tirages n'ont pas pu atteindre {res['n_lignes']} lignes à cause des limites ; leurs poids ont été remis à l'échelle.")
        notes.append("Les tirages respectent toujours la limite de 20 % par secteur, même si ton portefeuille la dépasse.")
        for nt in notes:
            st.caption(nt)

    # ------------------------------------------------------------- ÉTAPE 2
    st.markdown("### Étape 2 · Tester sur tes vrais rendements, sans supposer de loi normale")
    choix_serie = st.radio("Rendements testés", ["Portefeuille réel depuis le départ", "Portefeuille actuel rejoué sur un an"], horizontal=True,
                           help="Le portefeuille rejoué applique tes poids d'aujourd'hui aux cours de l'année passée : il montre le comportement de ta stratégie, "
                                "mais il ne prouve aucun talent, puisque tu as choisi ces lignes en connaissant déjà leur passé.")
    if choix_serie.startswith("Portefeuille réel"):
        sp = serie_portefeuille()
        premier_achat = tx.loc[tx["type"] == "achat", "date"].dropna()
        premier_achat = premier_achat.value_counts().idxmax() if len(premier_achat) else None
        r_test = sp["rend"].dropna() if sp else pd.Series(dtype=float)
        if premier_achat is not None:
            r_test = r_test[r_test.index > premier_achat]   # avant, le portefeuille était surtout en cash
    else:
        rr_ = adj_eur[[t for t in poids.index if t in adj_eur.columns]].pct_change(fill_method=None).iloc[-252:].fillna(0)
        r_test = rr_ @ poids.reindex(rr_.columns).fillna(0)
        st.warning("Biais de sélection : ces lignes ont été choisies en connaissant leur passé. Ce test décrit ta stratégie, il ne mesure pas ton talent.")
    r_bench = adj_eur[BENCH_PRINCIPAL].pct_change(fill_method=None).reindex(r_test.index) if BENCH_PRINCIPAL in adj_eur else None
    if len(r_test) < 20 or r_bench is None:
        st.info(f"Il faut au moins 20 jours de bourse pour cette étape ({len(r_test)} pour l'instant). En attendant, essaie « Portefeuille actuel rejoué sur un an ».")
    else:
        st.markdown("**a) Tes rendements suivent-ils une loi normale ?** (Heyman et al., 2014 ; Van Heerden, 2015)")
        nm = A.normalite(r_test)
        c = st.columns(4)
        kpi(c[0], "Asymétrie", f"{nm['asymetrie']:+.2f}".replace(".", ","), "0 pour une loi normale")
        kpi(c[1], "Kurtosis", f"{nm['kurtosis']:.2f}".replace(".", ","), "3 pour une loi normale")
        kpi(c[2], "Jarque-Bera, p", f"{nm['jb_p']:.3f}".replace(".", ","), "sous 0,05 : pas normale")
        kpi(c[3], "Shapiro-Wilk, p", f"{nm['sw_p']:.3f}".replace(".", ","), "sous 0,05 : pas normale")
        x = r_test.values
        fig = go.Figure(go.Histogram(x=x, nbinsx=40, histnorm="probability density", name="Tes rendements quotidiens", marker_color="#4c78a8", opacity=0.75))
        grille = np.linspace(x.min(), x.max(), 200)
        fig.add_trace(go.Scatter(x=grille, y=A.stats.norm.pdf(grille, x.mean(), x.std(ddof=1)), name="Loi normale de même moyenne et écart-type",
                                 line=dict(color=C_MAUVAIS, width=2)))
        fig.update_layout(height=280, xaxis_tickformat="+.1%", margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=1.15))
        graphe(fig, key="normalite")
        if min(nm["jb_p"], nm["sw_p"]) < 0.05:
            st.caption("La loi normale est rejetée : les tests classiques (t de Jensen, Sharpe) sont donc biaisés, d'où le bootstrap ci-dessous.")
        else:
            st.caption("La loi normale n'est pas rejetée sur cet échantillon, souvent parce qu'il est encore court. Le bootstrap reste plus sûr.")

        st.markdown("**b) Ton alpha résiste-t-il à un monde sans talent ?** (bootstrap de Kosowski et al., 2006, par blocs stationnaires)")
        bt = bootstrap_cache(r_test, r_bench)
        if bt:
            c = st.columns(4)
            kpi(c[0], "Alpha annualisé", pct(bt["alpha"] * 252, 1, True))
            kpi(c[1], "t de l'alpha", f"{bt['t_alpha']:+.2f}".replace(".", ","), f"bêta {bt['beta']:.2f}".replace(".", ","))
            kpi(c[2], "p-valeur classique (loi normale)", f"{bt['p_param']:.3f}".replace(".", ","))
            kpi(c[3], "p-valeur bootstrap", f"{bt['p_boot']:.3f}".replace(".", ","), "celle qu'il faut retenir")
            fig = go.Figure(go.Histogram(x=bt["t_boot"], nbinsx=60, marker_color="#c7c7c7", name="Monde sans talent"))
            fig.add_vline(x=bt["t_alpha"], line_color=C_MAUVAIS, line_width=2, annotation_text="toi", annotation_position="top")
            fig.add_vline(x=2, line_color=C_NEUTRE, line_dash="dot", annotation_text="t = 2", annotation_position="bottom right")
            fig.update_layout(height=260, margin=dict(t=30, l=0, r=0, b=0), xaxis_title="t de l'alpha dans 2 000 mondes sans talent", showlegend=False)
            graphe(fig, key="boot")
            st.caption("On garde tes vrais résidus et le vrai marché, on force l'alpha à zéro, puis on rééchantillonne par blocs de quelques jours pour garder "
                       "l'enchaînement des séances. Si ton t dépasse rarement ceux de ce monde, l'alpha est difficile à attribuer à la chance. "
                       "Avec beaucoup d'essais de stratégies, exige un t au-delà de 3 (Harvey, Liu et Zhu, 2016).")

        st.markdown("**c) Sharpe probabiliste** (Bailey et López de Prado, 2012)")
        ps = A.sharpe_probabiliste(r_test)
        if ps:
            c = st.columns(3)
            kpi(c[0], "Sharpe annualisé", f"{ps['sr_annuel']:.2f}".replace(".", ","))
            kpi(c[1], "Probabilité que le vrai Sharpe soit positif", pct(ps["psr"], 0))
            kpi(c[2], "Même calcul en supposant la loi normale", pct(ps["psr_normal"], 0), pct(ps["psr"] - ps["psr_normal"], 0, True))
            st.caption("L'écart entre les deux probabilités, c'est ce que coûte le fait d'ignorer l'asymétrie et les queues épaisses de tes rendements.")
            st.markdown("**Sharpe « dégonflé »** (Bailey et López de Prado, 2014)")
            n_ess = st.number_input("Nombre de versions du portefeuille que tu as essayées avant de garder celle-ci", 1, 200, 6, key="n_essais",
                                    help="Chaque version testée augmente la chance de tomber sur un bon Sharpe par hasard. Compte honnêtement : Excel, Colab, ajustements…")
            dsr = A.sharpe_degonfle(r_test, n_ess, RF())
            if dsr:
                c = st.columns(3)
                kpi(c[0], "Sharpe annualisé (net du taux sans risque)", f"{dsr['sr_annuel']:.2f}".replace(".", ","), delta_color="off")
                kpi(c[1], "Sharpe à battre vu le nombre d'essais", f"{dsr['sr_seuil_annuel']:.2f}".replace(".", ","), f"{n_ess} versions essayées", delta_color="off")
                kpi(c[2], "Probabilité d'un vrai Sharpe positif", pct(dsr["dsr"], 0), "après correction des essais multiples", delta_color="off")
                st.caption("Plus tu as essayé de versions, plus le Sharpe à battre est élevé : c'est la version « portefeuille » du problème des tests multiples de ton TFE.")

    st.markdown("**d) Portefeuilles jumeaux : battre des actions qui te ressemblent** (Daniel et al., 1997 ; Heyman et al., 2014)")
    if res and res.get("jumeaux"):
        jm = res["jumeaux"]
        c = st.columns(3)
        kpi(c[0], "Ton rendement", pct(jm["r_moi"], 2, True))
        kpi(c[1], "Mieux que tes jumeaux dans", pct(jm["percentile"], 0), "des 10 000 tirages")
        kpi(c[2], "p-valeur", f"{jm['p']:.2f}".replace(".", ","))
        fig = go.Figure(go.Histogram(x=jm["r_jumeaux"], nbinsx=70, marker_color="#c7c7c7"))
        fig.add_vline(x=jm["r_moi"], line_color=C_MAUVAIS, line_width=2, annotation_text="toi", annotation_position="top")
        fig.update_layout(height=240, xaxis_tickformat="+.1%", margin=dict(t=30, l=0, r=0, b=0), showlegend=False)
        graphe(fig, key="jumeaux")
        st.caption("Chaque ligne est remplacée par une action tirée au hasard du même secteur, du même tiers de momentum et du même tiers de volatilité. "
                   "Si tu bats tes jumeaux, la performance vient du choix des titres, pas d'une simple exposition au momentum, au secteur ou au risque.")
        det = jm["details"].copy()
        det["ticker"] = [pf["nom"].get(t, t) for t in det["ticker"]]
        with st.expander("Critères utilisés pour chaque ligne"):
            tableau(det, hide_index=True, column_config={"ticker": "Ligne", "jumeaux_possibles": "Jumeaux possibles", "critere": "Critères"})
    else:
        st.caption("Lance la simulation de l'étape 1 pour obtenir les jumeaux.")

    # ------------------------------------------------------------- ÉTAPE 3
    st.markdown("### Étape 3 · Combien de participants ont vraiment du talent ? (Barras, Scaillet et Wermers, 2010)")
    st.caption("Sur 200 participants, une dizaine paraîtront talentueux par pur hasard. Le taux de fausses découvertes estime la part de chance dans "
               "les « gagnants » à partir de la carte performance-risque d'APTimum (menu ≡ › télécharger en CSV ou Excel).")
    src_part = None
    chemin = os.path.join(ICI, "participants.csv")
    up = st.file_uploader("Fichier de la carte performance-risque", type=["csv", "xls", "xlsx"])
    if up is not None:
        src_part, err = lire_participants(up)
    elif os.path.exists(chemin):
        src_part, err = lire_participants(chemin)
    else:
        err = None
    if err:
        st.error(err)
    if src_part is not None and len(src_part.columns) >= 2:
        cols = list(src_part.columns)
        c = st.columns(5)
        c_nom = c[0].selectbox("Colonne du nom", cols, index=0)
        c_perf = c[1].selectbox("Colonne performance", cols, index=min(2, len(cols) - 1))
        c_risk = c[2].selectbox("Colonne risque", cols, index=min(1, len(cols) - 1))
        jours_p = c[3].number_input("Jours de bourse de la période", 1, 260, 20)
        ref = c[4].number_input("Performance de l'indice sur la période (%)", -50.0, 50.0, 0.0, step=0.1)
        dfp = pd.DataFrame({"nom": src_part[c_nom].astype(str), "performance": src_part[c_perf].map(num), "risque": src_part[c_risk].map(num)})
        for col in ["performance", "risque"]:
            if dfp[col].abs().max() > 1.5:
                dfp[col] = dfp[col] / 100
        fd = A.fdr_participants(dfp, jours_p, perf_ref=ref / 100)
        if not fd:
            st.warning("Il faut au moins 20 participants avec une performance et un risque.")
        else:
            c = st.columns(4)
            kpi(c[0], "Participants", fd["N"])
            kpi(c[1], "Sans talent (alpha nul)", pct(fd["pi0"], 0))
            kpi(c[2], "Vraiment meilleurs que l'indice", pct(fd["pi_plus"], 1))
            kpi(c[3], "Vraiment moins bons", pct(fd["pi_moins"], 1))
            st.caption(f"{fd['sig_pos']} participants paraissent significativement meilleurs au seuil de 10 %, mais on en attend environ "
                       f"{fd['faux_attendus']:.0f} par pur hasard : la part de chance parmi ces « gagnants » est estimée à {pct(fd['fdr_plus'], 0)}.")
            fig = go.Figure(go.Histogram(x=fd["table"]["t"], nbinsx=40, histnorm="probability density", marker_color="#4c78a8", opacity=0.75, name="Participants"))
            g_ = np.linspace(-4, 4, 200)
            fig.add_trace(go.Scatter(x=g_, y=A.stats.norm.pdf(g_), name="Si personne n'avait de talent", line=dict(color=C_MAUVAIS, width=2)))
            fig.update_layout(height=260, margin=dict(t=10, l=0, r=0, b=0), xaxis_title="t de chaque participant", legend=dict(orientation="h", y=1.15))
            graphe(fig, key="fdr")
            st.caption("Limites : un seul instantané, un t approximé à partir du risque annualisé, et des participants qui ne sont pas indépendants "
                       "(beaucoup détiennent les mêmes titres). C'est un ordre de grandeur, comme le rappelle ton TFE.")
    else:
        st.caption("Ajoute le fichier ici, ou dépose-le sur GitHub sous le nom participants.csv pour qu'il soit lu automatiquement.")

    # ------------------------------------------------------------- SIGNES EX ANTE
    st.markdown("### Les signes de talent mesurables à l'avance")
    st.caption("Ce qui, dans la littérature, prédit la performance future d'un gérant, calculé sur ton portefeuille actuel avec un an de cours.")
    r1 = adj_eur.pct_change(fill_method=None).iloc[-252:]
    lignes_ok = [t for t in poids.index if t in r1.columns]
    rp = (r1[lignes_ok].fillna(0) @ poids.reindex(lignes_ok).fillna(0))
    reg = A.regression(rp, r1[BENCH_PRINCIPAL]) if BENCH_PRINCIPAL in r1 else None
    enb = A.nombre_effectif_paris(poids, r1)
    c = st.columns(4)
    if reg:
        kpi(c[0], "R² ajusté face au marché", pct(reg["r2_adj"], 0), "plus bas = plus de sélection", delta_color="off")
        te = (rp - r1[BENCH_PRINCIPAL]).std() * np.sqrt(252)
        kpi(c[1], "Écart de suivi", pct(te, 1), "à quel point tu t'écartes de l'indice")
    if enb:
        kpi(c[2], "Nombre effectif de paris", f"{enb['enb']:.1f}".replace(".", ","), f"pour {enb['n_lignes']} lignes")
        kpi(c[3], "Part du premier facteur", pct(enb["part_premier_facteur"], 0), "le « marché » commun à tes lignes")
    st.markdown(
        "- **R² faible** : Amihud et Goyenko (2013) montrent que les fonds les moins expliqués par le marché et les facteurs surperforment ensuite.\n"
        "- **Écart à l'indice** : Cremers et Petajisto (2009) trouvent que les fonds qui s'écartent le plus de leur indice font mieux ; c'est contesté par "
        "Frazzini, Friedman et Pomorski (2016), qui l'attribuent aux petites capitalisations.\n"
        "- **Nombre effectif de paris** : le carré du ratio de diversification (Choueifaty et Coignard, 2008). 25 lignes qui bougent ensemble ne font que quelques paris.\n"
        "- **Meilleures idées** : Cohen, Polk et Silli (2010) montrent que les positions de plus forte conviction surperforment ; trop diluer détruit l'avantage.")

# ================================================================== JOURNAL
def page_journal():
    st.subheader("Journal des transactions")
    st.caption("Ajoute chaque ordre exécuté dans l'onglet « transactions » de ton Google Sheet : le site le prend en compte dans la minute.")
    tableau(tx.assign(date=tx["date"].dt.strftime("%d/%m/%Y")), hide_index=True,
                 column_config={"date": "Date", "ticker": "Ticker", "type": "Type", "quantite": "Quantité", "prix": "Prix", "frais": "Frais", "commentaire": "Commentaire"})
    st.subheader("Ventes réalisées")
    if ventes.empty:
        st.caption("Aucune vente pour l'instant.")
    else:
        tableau(ventes.assign(date=ventes["date"].dt.strftime("%d/%m/%Y")).style.format(
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
        kpi(c[0], "PGR (gains réalisés)", pct(pgr, 0))
        kpi(c[1], "PLR (pertes réalisées)", pct(plr, 0))
        kpi(c[2], "PGR − PLR", pct(pgr - plr, 0, True), "biais présent" if pgr - plr > 0.05 else "pas de biais net")


    st.markdown("### Prévisions et score de Brier")
    intro("Avant chaque décision, écris une prévision chiffrée dans previsions.csv : par exemple « Ferrari fera mieux que l'Euronext 100 d'ici le 30/11 », "
          "avec ta probabilité (0,6 pour 60 %). À l'échéance, note le résultat : 1 si c'est arrivé, 0 sinon. Le score de Brier mesure la qualité de ton jugement, "
          "indépendamment de la chance d'un résultat isolé (Brier, 1950 ; Tetlock et Gardner, 2015).")
    try:
        pv = pd.read_csv(os.path.join(ICI, "previsions.csv"), dtype=str)
    except Exception:
        pv = pd.DataFrame()
    if pv.empty:
        st.caption("Aucune prévision pour l'instant. Format d'une ligne : 2026-11-05,Ferrari,Fera mieux que l'Euronext 100,0.6,2026-11-30, "
                   "(laisse le résultat vide jusqu'à l'échéance, puis mets 1 ou 0).")
    else:
        pv["probabilite"] = pv["probabilite"].map(num)
        pv["resultat"] = pv["resultat"].map(num)
        tableau(pv.style.format({"probabilite": lambda v: pct(v, 0), "resultat": lambda v: "—" if pd.isna(v) else ("oui" if v == 1 else "non")}),
                hide_index=True, column_config={"date": "Date", "ligne": "Ligne", "prevision": "Prévision", "probabilite": "Probabilité",
                                                "echeance": "Échéance", "resultat": "Arrivé ?"})
        sb = A.score_brier(pv)
        if sb:
            c = st.columns(4)
            kpi(c[0], "Prévisions résolues", sb["n"])
            kpi(c[1], "Score de Brier", f"{sb['brier']:.3f}".replace(".", ","), "0 = parfait, 0,25 = pile ou face")
            kpi(c[2], "Taux de réussite", pct(sb["taux_reussite"], 0))
            kpi(c[3], "Gain sur une prévision naïve", pct(sb["skill"], 0), "au-dessus de 0 : tu apportes de l'information")
            cal = sb["calibration"]
            if len(cal) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#9aa4b2"), name="Calibration parfaite"))
                fig.add_trace(go.Scatter(x=cal["prevu"], y=cal["realise"], mode="markers+lines", marker=dict(size=10 + 2 * cal["n"]), name="Toi"))
                fig.update_layout(height=300, xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="probabilité annoncée", yaxis_title="fréquence réelle",
                                  margin=dict(t=10, l=0, r=0, b=0))
                graphe(fig, key="calib")

# ================================================================== DONNÉES
def page_donnees():
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
    tableau(q.style.format({"saut_max_30j": lambda v: pct(v, 1)}).map(lambda v: "" if v == "OK" else f"color:{C_ATTENTION};font-weight:600", subset=["etat"]),
                 hide_index=True, column_config={"ticker": "Ticker", "dernier_jour": "Dernier cours", "jours": "Jours d'historique", "saut_max_30j": "Plus fort mouvement sur 30 jours", "etat": "État"})
    st.caption("Un « saut suspect » peut être une vraie nouvelle (résultats) ou une erreur d'ajustement de Yahoo : compare avec NextWise.")
    st.caption("Devises converties en euros : " + (", ".join(f"{t} ({d})" for t, d in devises.items() if d != "EUR") or "aucune"))


# ================================================================== OPTIMISATION
@st.cache_data(ttl=3600, show_spinner="Optimisation…")
def optim_cache(R, borne, investi):
    cov = R.cov().values * 252
    res = {}
    for mode in ["minvar", "mdp", "erc"]:
        res[mode] = A.optimiser(cov, mode, borne) * investi
    return res, cov


def page_optimisation():
    st.subheader("Optimisation")
    intro("Ce que donneraient tes 25 lignes avec d'autres règles de pondération. Les trois portefeuilles optimisés n'utilisent <b>que le risque</b> "
          "(volatilités et corrélations), jamais de prévision de rendement : c'est volontaire. Les optimisations qui prévoient les rendements sont si "
          "sensibles aux erreurs d'estimation qu'elles font souvent moins bien que les poids égaux (Michaud, 1989 ; DeMiguel, Garlappi et Uppal, 2009).")
    t_ok = [t for t in poids.index if t in adj_eur.columns and adj_eur[t].iloc[-252:].notna().sum() > 200]
    R = adj_eur[t_ok].pct_change(fill_method=None).iloc[-252:].fillna(0)
    investi = float(poids.reindex(t_ok).sum())
    borne = st.slider("Poids maximum par ligne", 4.0, 10.0, 6.0, 0.5, format="%.1f %%",
                      help="10 % est la limite UCITS. Plus bas, les portefeuilles restent proches des poids égaux.") / 100
    res, cov = optim_cache(R, borne / max(investi, 1e-9), investi)
    sig = np.sqrt(np.diag(cov))
    mu = (adj_eur[t_ok].iloc[-1] / adj_eur[t_ok].iloc[-253] - 1).values
    rb = adj_eur[BENCH_PRINCIPAL].pct_change(fill_method=None).iloc[-252:]
    portefeuilles = {"Ton portefeuille": poids.reindex(t_ok).fillna(0).values, "Poids égaux": np.full(len(t_ok), investi / len(t_ok)),
                     "Variance minimale": res["minvar"], "Diversification maximale": res["mdp"], "Parité de risque": res["erc"]}
    lignes_t = []
    for nom_p, w in portefeuilles.items():
        v = float(np.sqrt(w @ cov @ w))
        dr = float((w * sig).sum() / v) if v > 0 else np.nan
        rp_ = R.values @ w
        beta = np.cov(rp_, rb.fillna(0).values)[0, 1] / rb.var() if rb.var() > 0 else np.nan
        sect = pd.Series(w, index=t_ok).groupby(pf["secteur"].reindex(t_ok).fillna("?")).sum()
        lignes_t.append({"portefeuille": nom_p, "vol": v, "dr": dr, "paris": dr ** 2, "beta": beta, "var95": -np.percentile(rp_, 5),
                         "max_ligne": w.max(), "max_secteur": sect.max(), "secteur_max": sect.idxmax(), "rend_passe": float(w @ mu)})
    tb = pd.DataFrame(lignes_t)
    tableau(tb.style.format({"vol": lambda v: pct(v), "dr": lambda v: f"{v:.2f}".replace(".", ","), "paris": lambda v: f"{v:.1f}".replace(".", ","),
                             "beta": lambda v: f"{v:.2f}".replace(".", ","), "var95": lambda v: pct(v, 2), "max_ligne": lambda v: pct(v),
                             "max_secteur": lambda v: pct(v), "rend_passe": lambda v: pct(v, 1, True)})
            .map(lambda v: f"color:{C_MAUVAIS};font-weight:600" if isinstance(v, float) and v > LIM["secteur"] + 1e-9 else "", subset=["max_secteur"]),
            hide_index=True,
            column_config={"portefeuille": "Pondération", "vol": "Volatilité prévue", "dr": "Ratio de diversification", "paris": "Paris indépendants",
                           "beta": "Bêta", "var95": "VaR 95 % (1 jour)", "max_ligne": "Plus grosse ligne", "max_secteur": "Premier secteur",
                           "secteur_max": "Lequel", "rend_passe": "Rendement passé 1 an"})
    st.caption("Le rendement passé est affiché pour information : il ne dit rien du futur et ne sert pas au calcul. En rouge : un secteur au-dessus de ta limite de 20 %.")

    st.markdown("### Les poids, ligne par ligne")
    wdf = pd.DataFrame({k_: v for k_, v in portefeuilles.items() if k_ != "Poids égaux"}, index=[pf["nom"].get(t, t) for t in t_ok])
    wdf = wdf.sort_values("Ton portefeuille")
    fig = go.Figure()
    for i_, nom_ in enumerate(wdf.index):
        fig.add_trace(go.Scatter(x=[wdf.loc[nom_].min(), wdf.loc[nom_].max()], y=[nom_, nom_], mode="lines", line=dict(color="rgba(127,127,127,.35)", width=3),
                                 showlegend=False, hoverinfo="skip"))
    symb = {"Ton portefeuille": "circle", "Variance minimale": "square", "Diversification maximale": "diamond", "Parité de risque": "triangle-up"}
    for i, c in enumerate(wdf.columns):
        fig.add_trace(go.Scatter(x=wdf[c], y=wdf.index, mode="markers", name=c, marker=dict(size=11 if i == 0 else 9, color=PALETTE[i], symbol=symb.get(c, "circle"),
                                 line=dict(width=1, color="white")), hovertemplate="%{y} · " + c + " : %{x:.2%}<extra></extra>"))
    fig.add_vline(x=borne, line_dash="dot", line_color="#9aa4b2", annotation_text="poids max")
    fig.update_layout(height=max(420, 26 * len(wdf)), xaxis_tickformat=".1%", margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=1.04))
    graphe(fig, key="optim_poids")

    with st.expander("Pour aller plus loin : le nuage des portefeuilles possibles, en 3D"):
      intro("4 000 pondérations tirées au hasard de tes 25 lignes, plafonnées au poids maximum. Axes : volatilité prévue, ratio de diversification, rendement passé "
            "sur un an. Couleur : Sharpe passé. Tourne le graphique à la souris. Les losanges sont les cinq pondérations du tableau.")
      nu = A.nuage_portefeuilles(cov, mu, n=4000, borne=borne / max(investi, 1e-9))
      fig = go.Figure()
      fig.add_trace(go.Scatter3d(x=nu["vol"] * investi, y=nu["dr"], z=nu["rendement"] * investi, mode="markers",
                                 marker=dict(size=2.2, color=nu["sharpe"], colorscale="Viridis", opacity=0.55, colorbar=dict(title="Sharpe passé")),
                                 name="Aléatoires", hovertemplate="vol %{x:.1%}<br>diversification %{y:.2f}<br>rendement passé %{z:+.1%}<extra></extra>"))
      for i, r in tb.iterrows():
          fig.add_trace(go.Scatter3d(x=[r["vol"]], y=[r["dr"]], z=[r["rend_passe"]], mode="markers+text", text=[r["portefeuille"]],
                                     marker=dict(size=7, symbol="diamond", color=PALETTE[i % len(PALETTE)]), name=r["portefeuille"]))
      fig.update_layout(height=620, margin=dict(t=10, l=0, r=0, b=0), showlegend=False,
                        scene=dict(xaxis=dict(title="volatilité", tickformat=".0%"), yaxis=dict(title="diversification"), zaxis=dict(title="rendement passé", tickformat="+.0%")))
      graphe(fig, key="nuage3d")
      st.caption("À lire : la diversification maximale cherche le haut de l'axe « diversification », la variance minimale le bord gauche de l'axe « volatilité ». "
                 "Si ton portefeuille est loin des deux, c'est que tes choix de poids (convictions, règles de volatilité) coûtent de la diversification : à toi de juger si c'est voulu.")


# ================================================================== VALORISATION
def page_valorisation():
    st.subheader("Valorisation")
    intro("Ce que tu paies pour la croissance attendue. Les règles : PEG ≤ 2 pour les valeurs de croissance (Lynch), ratio de Lynch ≥ 1 pour les croissances "
          "lentes à dividende ((croissance + rendement) ÷ PER). Les cycliques ont un PER trompeur (bas au sommet du cycle) et les financières se jugent sur "
          "le ROE et la valeur comptable : elles sont donc traitées à part.")
    v = pf[["nom", "type_valo", "per", "croissance", "peg", "lynch", "roe", "diagnostic", "verdict", "poids"]].copy()
    pts = v.dropna(subset=["per", "croissance"])
    if pts.empty:
        st.info("Ajoute PER et croissance dans lignes.csv.")
        return
    coul = {"Valorisation OK": PALETTE[0], "Cas 1": "#F59E0B", "Cas 2": "#F97316", "Cas 3": "#EF4444"}
    fig = go.Figure()
    xmax = max(0.5, pts["croissance"].max() * 1.1)
    gg = np.linspace(0.001, xmax, 50)
    for peg_, dash in [(1, "dot"), (2, "dash")]:
        fig.add_trace(go.Scatter(x=gg, y=peg_ * gg * 100, mode="lines", line=dict(color="#9aa4b2", dash=dash, width=1), name=f"PEG {peg_}", hoverinfo="skip"))
    for diag, grp in pts.groupby(pts["diagnostic"].fillna("Valorisation OK")):
        etiq = [n if (diag != "Valorisation OK" or (pd.notna(p_) and p_ > 2)) else "" for n, p_ in zip(grp["nom"], grp["peg"])]
        fig.add_trace(go.Scatter(x=grp["croissance"], y=grp["per"], mode="markers+text", text=etiq, textposition="top center", textfont=dict(size=11),
                                 marker=dict(size=10 + 120 * grp["poids"].fillna(0.03), color=coul.get(diag, "#64748B"), line=dict(width=1, color="white")),
                                 name=diag, customdata=np.stack([grp["type_valo"].fillna(""), grp["peg"].fillna(np.nan), grp["lynch"].fillna(np.nan)], axis=-1),
                                 hovertext=grp["nom"], hovertemplate="%{hovertext}<br>PER %{y:.1f} · croissance %{x:+.0%}<br>%{customdata[0]} · PEG %{customdata[1]:.2f} · Lynch %{customdata[2]:.2f}<extra></extra>"))
    fig.update_layout(height=520, xaxis_tickformat="+.0%", xaxis_title="croissance attendue du bénéfice par action", yaxis_title="PER",
                      yaxis_range=[0, max(40, pts["per"].max() * 1.1)], margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=1.05))
    graphe(fig, key="valo")
    st.caption("Au-dessus de la droite PEG 2, la croissance ne justifie plus le prix. La taille des points suit le poids de la ligne. Données MarketScreener "
               "saisies dans lignes.csv : mets-les à jour après chaque publication de résultats.")
    st.markdown("### Les diagnostics")
    st.markdown("- **Cas 1** : au-dessus du seuil, mais justifié par la qualité et la visibilité (contrats longs, position dominante).\n"
                "- **Cas 2** : signal de valorisation à analyser : vrai problème ou artefact des données ?\n"
                "- **Cas 3** : hors critères, décision ou exception écrite obligatoire.")
    sig_ = v[v["diagnostic"].fillna("").str.startswith("Cas")].sort_values("diagnostic", ascending=False)
    if len(sig_):
        tableau(sig_[["nom", "diagnostic", "type_valo", "per", "croissance", "peg", "lynch", "verdict"]].style.format(
            {"per": lambda x: "—" if pd.isna(x) else f"{x:.1f}".replace(".", ","), "croissance": lambda x: pct(x, 0, True),
             "peg": lambda x: "—" if pd.isna(x) else f"{x:.2f}".replace(".", ","), "lynch": lambda x: "—" if pd.isna(x) else f"{x:.2f}".replace(".", ",")}),
            hide_index=True, column_config={"nom": "Ligne", "diagnostic": "Cas", "type_valo": "Type", "per": "PER", "croissance": "Croissance",
                                            "peg": "PEG", "lynch": "Lynch", "verdict": "Décision"})
    st.markdown("### Toutes les lignes")
    tableau(v.sort_values("type_valo")[["nom", "type_valo", "per", "croissance", "peg", "lynch", "roe", "diagnostic"]].style.format(
        {"per": lambda x: "—" if pd.isna(x) else f"{x:.1f}".replace(".", ","), "croissance": lambda x: pct(x, 0, True), "roe": lambda x: pct(x, 0),
         "peg": lambda x: "—" if pd.isna(x) else f"{x:.2f}".replace(".", ","), "lynch": lambda x: "—" if pd.isna(x) else f"{x:.2f}".replace(".", ",")}),
        hide_index=True, column_config={"nom": "Ligne", "type_valo": "Type", "per": "PER", "croissance": "Croissance", "peg": "PEG", "lynch": "Lynch",
                                        "roe": "ROE", "diagnostic": "Diagnostic"})


# ================================================================== SCORE DU CONCOURS
GICS_11 = ["Énergie", "Matériaux", "Industrie", "Conso discrétionnaire", "Conso de base", "Santé", "Finance", "Technologie",
           "Services de communication", "Utilities", "Immobilier"]


def page_score():
    st.subheader("Score du concours")
    intro("Le règlement note sept critères : performance brute, performance ajustée au risque, HHI, nombre de secteurs, ratio de diversification, respect "
          "d'UCITS et quiz. Leur pondération n'est pas publiée. Quatre d'entre eux ne dépendent pas de la chance : c'est là que se gagnent les points sûrs.")
    w_inv = poids.dropna()
    h, n_eff = A.hhi(w_inv)
    n = len(w_inv)
    secteurs = sorted(set(pf["secteur"].dropna()) & set(GICS_11))
    manquants = [x for x in GICS_11 if x not in secteurs]
    ucits_ok = all(cons.iloc[:2]["etat"] != "Dépassée") if len(cons) >= 2 else True
    vol_p = risque["vol"] if risque else np.nan
    ratio_pr = perf / vol_p if pd.notna(vol_p) and vol_p > 0 else np.nan

    st.markdown("### Les critères que tu maîtrises")
    c = st.columns(4)
    kpi(c[0], "HHI (concentration)", f"{h:.3f}".replace(".", ","), f"minimum possible avec {n} lignes : {1 / n:.3f}".replace(".", ","), delta_color="off")
    kpi(c[1], "Nombre effectif de lignes", f"{n_eff:.1f}".replace(".", ","), f"sur {n} lignes détenues", delta_color="off")
    kpi(c[2], "Secteurs couverts", f"{len(secteurs)} / 11", "manque : " + ", ".join(manquants) if manquants else "tous couverts", delta_color="off")
    kpi(c[3], "UCITS 5/10/40", "respecté" if ucits_ok else "NON respecté", f"plus grosse ligne {pct(w_inv.max())}", delta_color="off")
    c = st.columns(4)
    kpi(c[0], "Ratio de diversification", f"{risque['dr']:.2f}".replace(".", ",") if risque else "—", "Σ poids × vol ÷ vol du portefeuille", delta_color="off")
    kpi(c[1], "Performance brute", pct(perf, 2, True), eur(total - apports))
    kpi(c[2], "Performance ÷ risque prévu", f"{ratio_pr:.2f}".replace(".", ",") if pd.notna(ratio_pr) else "—",
        "la pente de la carte APTimum (droite rouge à 0,3)", delta_color="off")
    kpi(c[3], "Quiz", "à faire", "points sans risque : fais-les tous", delta_color="off")

    st.markdown("### Ce qui ferait gagner des points")
    actions = []
    if manquants:
        univ = charger_univers()
        for sct in manquants:
            noms_c = univ[univ["secteur"] == sct]["nom"].head(5).tolist()
            actions.append(f"**+1 secteur** : ajoute une ligne **{sct}**" + (f" (par exemple {', '.join(noms_c)}) ; vérifie-la dans Candidats." if noms_c else "."))
    if n and h > 1.08 / n:
        actions.append(f"**HHI** : à poids égaux, ton HHI passerait de {h:.4f} à {1 / n:.4f}. L'écart vient surtout des lignes à 2,5 % (règle de volatilité) et des écarts de cours ; "
                       "c'est un coût voulu de ta règle de risque, à garder en tête.".replace(".", ",", 2))
    if risque:
        t_ok = [t for t in poids.index if t in adj_eur.columns and adj_eur[t].iloc[-252:].notna().sum() > 200]
        R = adj_eur[t_ok].pct_change(fill_method=None).iloc[-252:].fillna(0)
        inv = float(poids.reindex(t_ok).sum())
        res_o, cov_o = optim_cache(R, 0.06 / max(inv, 1e-9), inv)
        w_mdp = res_o["mdp"]
        sig = np.sqrt(np.diag(cov_o))
        dr_mdp = float((w_mdp * sig).sum() / np.sqrt(w_mdp @ cov_o @ w_mdp))
        if dr_mdp > risque["dr"] + 0.03:
            actions.append(f"**Ratio de diversification** : la pondération à diversification maximale (plafond 6 % par ligne) le ferait passer de "
                           f"{risque['dr']:.2f} à {dr_mdp:.2f}. Voir la page Optimisation pour les poids ; à arbitrer avec tes convictions.".replace(".", ",", 2))
    tx_cash = cash / total if total else 0
    if tx_cash > 0.03:
        actions.append(f"**Cash** : {pct(tx_cash)} ne travaille pas. Le concours couvre novembre-avril, la période historiquement la plus favorable aux actions "
                       "(Jacobsen et Zhang, 2018) : un cash élevé coûte de la performance brute.")
    for _, c_ in cons.iterrows():
        if c_["etat"] != "Respectée":
            actions.append(f"**Contrainte {c_['etat'].lower()}** : {c_['regle']} à {pct(c_['niveau'])} pour {pct(c_['limite'], 0)}. "
                           "Si le jury vérifie les mêmes limites que toi, c'est un point d'exécution à corriger au rééquilibrage.")
    actions.append("**Quiz** : ils comptent dans le classement et ne dépendent que de toi. Fais-les tous, dès qu'ils sortent.")
    for a_ in actions:
        st.markdown("- " + a_)

    st.markdown("### Contribution de chaque ligne à la concentration")
    wn = (w_inv / w_inv.sum()).sort_values(ascending=False)
    cont = (wn ** 2) / (wn ** 2).sum()
    fig = go.Figure(go.Bar(x=[pf["nom"].get(t, t) for t in wn.index], y=wn.values, marker_color=[C_MAUVAIS if v > 1.2 / n else PALETTE[0] for v in wn.values],
                           text=[pct(v) for v in wn.values], textposition="outside", customdata=cont.values,
                           hovertemplate="%{x}<br>poids %{y:.1%}<br>part du HHI %{customdata:.1%}<extra></extra>"))
    fig.add_hline(y=1 / n, line_dash="dot", line_color="#9aa4b2", annotation_text="poids égal")
    fig.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(t=20, l=0, r=0, b=0))
    graphe(fig, key="score_hhi")
    st.caption("Les poids sont ramenés au capital investi, comme pour un HHI classique. En rouge : les lignes nettement au-dessus du poids égal.")

    st.markdown("### Ce qu'il faut demander aux organisateurs")
    st.code("Bonjour,\n\nPour le classement quantitatif du NextWise Challenge, pourriez-vous préciser :\n"
            "1. la pondération des sept critères (performance brute, ajustée au risque, HHI, secteurs, ratio de diversification, UCITS, quiz) ;\n"
            "2. la mesure du risque utilisée pour la performance ajustée au risque (volatilité réalisée sur la période ou risque ex ante APTimum, et à quelle date) ;\n"
            "3. si les dividendes sont crédités sur les portefeuilles.\n\nMerci d'avance,\nHugo Lefebvre", language=None)
    st.caption("Les réponses changent la stratégie : par exemple, un risque mesuré ex ante à la date de clôture favoriserait une réduction du risque en fin de concours.")


# ================================================================== RAPPORT
def page_rapport():
    st.html("""<style>@media print { [data-testid="stSidebar"], header, [data-testid="stToolbar"] {display:none !important;}
    .block-container {padding-top: 0 !important; max-width: 100% !important;} }</style>""")
    st.subheader(f"Rapport au {dt.datetime.now(TZ):%d/%m/%Y}")
    intro("Une page qui résume le portefeuille, pensée pour le jury ou pour ton bilan du mois. Pour l'enregistrer en PDF : "
          "<b>Ctrl + P</b> › « Enregistrer au format PDF » (le menu de gauche disparaît à l'impression).")
    sp = serie_portefeuille()
    c = st.columns(4)
    kpi(c[0], "Performance depuis le départ", pct(perf, 2, True), eur(total - apports))
    if sp:
        dd = (sp["indice"] / sp["indice"].cummax() - 1).min()
        kpi(c[1], "Plus forte baisse", pct(dd, 1), "depuis un sommet", delta_color="off")
        jours_ = sp["jours"]
        bs = close_eur[BENCH_PRINCIPAL].reindex(jours_).ffill() if BENCH_PRINCIPAL in close_eur else None
        if bs is not None and len(bs.dropna()):
            kpi(c[2], "Euronext 100 sur la période", pct(bs.iloc[-1] / bs.dropna().iloc[0] - 1, 2, True), f"depuis le {jours_[0]:%d/%m}", delta_color="off")
    n_ko = int((regles_lignes["ecarts"] > 0).sum())
    kpi(c[3], "Lignes conformes à toutes les règles", f"{len(regles_lignes) - n_ko} / {len(regles_lignes)}", delta_color="off")
    g, d = st.columns([3, 2])
    with g:
        if sp:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sp["indice"].index, y=sp["indice"] - 1, name="Portefeuille", line=dict(width=2.5)))
            for bt, bn in BENCH.items():
                if bt in close_eur:
                    b_ = close_eur[bt].reindex(sp["jours"]).ffill()
                    fig.add_trace(go.Scatter(x=b_.index, y=b_ / b_.dropna().iloc[0] - 1, name=bn, line=dict(width=1.5, dash="dot")))
            fig.update_layout(height=320, yaxis_tickformat="+.1%", margin=dict(t=30, l=0, r=0, b=0), legend=dict(orientation="h", y=-0.15),
                              title=dict(text="Performance face aux indices", font_size=13))
            graphe(fig, key="rap_perf")
    with d:
        st.markdown("**Risque**")
        if risque:
            c2 = st.columns(2)
            kpi(c2[0], "Volatilité prévue", pct(risque["vol"]), f"bêta {risque['beta']:.2f}".replace(".", ","), delta_color="off")
            kpi(c2[1], "Paris indépendants", f"{risque['dr'] ** 2:.1f}".replace(".", ","), f"ratio de diversification {risque['dr']:.2f}".replace(".", ","), delta_color="off")
        st.markdown("**Contraintes**")
        jauges(cons)
    g, d = st.columns(2)
    with g:
        sec = poids.groupby(pf["secteur"].fillna("?")).sum().sort_values()
        fig = go.Figure(go.Bar(x=sec.values, y=sec.index, orientation="h", marker_color=PALETTE[0], text=[pct(v) for v in sec.values], textposition="outside"))
        fig.add_vline(x=LIM["secteur"], line_dash="dash", line_color=C_MAUVAIS, annotation_text="limite")
        fig.update_layout(height=320, xaxis_tickformat=".0%", margin=dict(t=30, l=0, r=0, b=0), title=dict(text="Répartition par secteur", font_size=13))
        graphe(fig, key="rap_sect")
    with d:
        pc = pf.groupby(pf["poche"].fillna("Sans poche"))["poids"].sum().sort_values()
        fig = go.Figure(go.Bar(x=pc.values, y=pc.index, orientation="h", marker_color=PALETTE[5], text=[pct(v) for v in pc.values], textposition="outside"))
        fig.update_layout(height=320, xaxis_tickformat=".0%", margin=dict(t=30, l=0, r=0, b=0), title=dict(text="Répartition par poche (thèse)", font_size=13))
        graphe(fig, key="rap_poche")
    st.markdown("### Les points à retenir")
    pts = []
    mx = pf["latent"].idxmax() if pf["latent"].notna().any() else None
    mn = pf["latent"].idxmin() if pf["latent"].notna().any() else None
    if mx is not None:
        pts.append(f"Meilleure contribution : **{pf.loc[mx, 'nom']}** ({eur(pf.loc[mx, 'latent'])}) ; moins bonne : **{pf.loc[mn, 'nom']}** ({eur(pf.loc[mn, 'latent'])}).")
    dep = cons[cons["etat"] != "Respectée"]
    pts.append("Contraintes : " + ("toutes respectées." if dep.empty else ", ".join(f"{r['regle'].lower()} ({r['etat'].lower()}, {pct(r['niveau'])})" for _, r in dep.iterrows()) + "."))
    if risque:
        top_r = risque["contrib"].sort_values(ascending=False).head(3)
        pts.append("Plus fortes contributions au risque : " + ", ".join(f"{pf['nom'].get(t, t)} ({pct(v)})" for t, v in top_r.items()) + ".")
    if n_ko:
        pts.append(f"{n_ko} lignes s'écartent d'au moins une règle d'entrée : voir la page Règles et le test de la page blanche.")
    for p_ in pts:
        st.markdown("- " + p_)
    st.markdown("### Décisions des 30 derniers jours")
    recents = tx[(tx["date"] >= pd.Timestamp.now() - pd.Timedelta(days=30)) & tx["type"].isin(["achat", "vente"])]
    if recents.empty:
        st.caption("Aucun ordre sur les 30 derniers jours.")
    else:
        rr_ = recents.assign(date=recents["date"].dt.strftime("%d/%m/%Y"), type=recents["type"].str.capitalize(), nom=[pf["nom"].get(t, lignes["nom"].get(t, t) if t in lignes.index else t) for t in recents["ticker"]])
        tableau(rr_[["date", "nom", "type", "quantite", "prix"]].style.format({"quantite": "{:g}", "prix": lambda v: eur(v, 2)}), hide_index=True,
                column_config={"date": "Date", "nom": "Ligne", "type": "Ordre", "quantite": "Quantité", "prix": "Prix"})
    st.caption("Sources : NextWise (transactions), Yahoo Finance (cours différés), MarketScreener (fondamentaux), APTimum (scénarios). "
               "Les chiffres officiels du concours restent ceux de NextWise.")


# ================================================================== GUIDE
GUIDE = {
    "Ta routine": """
- **Chaque jour, 2 minutes** : le *Tableau de bord*. Regarde les alertes rouges et les contraintes, puis la carte des lignes.
- **Avant chaque achat** : *Candidats* pour trouver, *Valorisation* et MarketScreener pour vérifier, *Simulateur* pour tester, puis écris la thèse, la menace et la condition de sortie dans lignes.csv.
- **Le 1er de chaque mois** : *Rééquilibrage* (ordres et test de la page blanche), *Règles*, *Risque*, *Stress tests*, puis *Chance ou talent* sur le mois écoulé. Note trois chiffres : ton percentile à risque égal, ton ratio de diversification, ta plus forte baisse. Termine par le *Rapport*, à enregistrer en PDF.
- **Après chaque ordre** : ajoute la ligne dans transactions.csv (date AAAA-MM-JJ, point décimal, prix en euros = montant NextWise ÷ quantité).
""",
    "Score du concours": """
- Les quatre critères sans chance : **HHI** (plus bas = moins concentré ; le minimum avec N lignes à poids égaux est 1/N), **secteurs couverts sur 11**, **ratio de diversification**, **UCITS**. Plus les **quiz**.
- La liste « ce qui ferait gagner des points » est recalculée à chaque ouverture : secteur manquant, écart à la diversification maximale, cash qui ne travaille pas, contrainte dépassée.
- Le message tout prêt pour les organisateurs : pondération des critères, mesure du risque, dividendes.
""",
    "Tableau de bord": """
- **Bandeau** : valeur totale et variation du jour, gain du jour face à l'Euronext 100, performance depuis le départ (même calcul que NextWise), cash, volatilité prévue, bêta, ratio de diversification.
- **Carte** : taille = poids, couleur = variation du jour, regroupée par poche. Toute une poche de la même couleur = mouvement de thème, pas d'entreprise.
- **Alertes** : rouge = une règle est touchée, orange = à surveiller. Une alerte n'est pas un ordre : c'est le signal de relire la fiche de la ligne.
- **Contraintes** : UCITS 5/10/40, secteur ≤ 20 %, place ≤ 30 %. « Proche » = moins de 2 à 3 points de marge.
""",
    "Lignes et journal": """
- **Prix de revient** : prix d'achat moyen. Il est affiché pour information : il n'entre dans aucune décision (effet de disposition).
- **Momentum 12-1** : sous −20 %, c'est une règle de sortie. **Vol EWMA** : au-dessus de 40 %, poids réduit à 2,5 %.
- **Part du risque** : à comparer au poids. Une ligne qui porte bien plus de risque que de capital concentre le danger.
- **Journal** : PGR − PLR au-dessus de 5 points = tu vends tes gagnants trop tôt. **Prévisions** : le score de Brier mesure ton jugement, pas ta chance.
""",
    "Candidats, simulateur, rééquilibrage, règles": """
- **Candidats** : d'abord « aucun écart aux règles », puis Δ diversification positif et Δ volatilité négatif, puis le meilleur score de momentum. Le mode « remplacement » répond à « par quoi remplacer Bureau Veritas ? ».
- **Simulateur** : modifie les poids, mets 0 pour une vente. Si l'ordre fait dépasser une contrainte ou baisser la diversification sans raison, ne le passe pas.
- **Rééquilibrage** : la bande de tolérance évite de trader pour quelques euros. Copie les lignes proposées dans le journal après exécution, avec les vrais prix.
- **Règles** : ✓ / ✗ par ligne. Une ligne détenue qui perd une règle d'entrée se réexamine au rééquilibrage, elle ne se vend pas dans la minute.
""",
    "Risque, stress, régime, optimisation": """
- **Volatilité prévue** : un an de cours. **Ratio de diversification** : plus il est haut, mieux tes lignes se compensent ; son carré = nombre de paris indépendants.
- **VaR 95 %** : un jour sur vingt, tu perds plus que ça. **ES** : la perte moyenne ces jours-là.
- **Facteurs** : bêta significatif = pari sur ce facteur. R² faible = beaucoup de sélection propre.
- **Stress tests** : crises rejouées sur les vrais cours, scénarios APTimum, curseurs. Les curseurs sous-estiment les vraies crises (les corrélations montent).
- **Régime** : turbulence au-dessus du 95e centile ou ratio d'absorption en forte hausse = marché fragile ; vérifie ton cash et tes stress tests.
- **Optimisation** : compare ta pondération à la variance minimale, à la diversification maximale et à la parité de risque, sans prévision de rendement.
""",
    "Performance, momentum, valorisation, chance ou talent": """
- **Performance** : courbe pondérée par le temps (les apports ne comptent pas comme gain). **Ratios** : Sharpe corrigé de Lo, Sortino, ratio d'information, M² (le plus lisible : ta performance au risque de l'indice), captures haussière et baissière, VaR de Cornish-Fisher ; chaque ligne dit si tu fais mieux ou moins bien que l'indice. **Brinson** : allocation = bons secteurs ; sélection = bons titres dans le secteur ; la ligne « Cash » mesure le coût de ne pas être investi.
- **Momentum** : le score combine momentum résiduel, 12-7, régularité et plus haut 52 semaines. Orange sur « dernier mois » = risque d'inversion. L'IC dit si le signal marche sur ton univers.
- **Valorisation** : au-dessus de la droite PEG 2, la croissance ne justifie plus le prix. Cas 1, 2, 3 = du plus défendable au plus exigeant.
- **Chance ou talent** : suit ton TFE. Monte Carlo (percentile à risque égal), bootstrap (p-valeur sans loi normale), jumeaux (bats-tu des titres qui te ressemblent ?), FDR (part de talent parmi tous les participants). Une période ne prouve rien : c'est la répétition qui compte.
""",
    "Rapport": """
- Une page de synthèse pour le jury ou ton bilan mensuel : performance face aux indices, risque, contraintes, répartition, points à retenir, ordres du mois.
- **Ctrl + P › Enregistrer au format PDF** pour en faire un document (le menu disparaît à l'impression).
""",
    "Données et limites": """
- **Cours** : Yahoo Finance, différés d'environ 15 minutes, convertis en euros. La référence du concours reste NextWise.
- **Onglet Données** : « saut suspect » = vérifie sur NextWise ; « cours ancien » = ticker à vérifier.
- **Fondamentaux** : lignes.csv (MarketScreener) pour tes lignes, Yahoo (indicatif) pour les candidats.
- **Affichage** : si un élément ne charge pas, le site se répare seul en quelques secondes et revient sur la même page.
""",
}


def page_guide():
    st.subheader("Guide")
    intro("Comment lire chaque page, et la routine qui va avec. Toutes les références citées sur le site sont des articles publiés ; les noms et années "
          "te permettent de les retrouver sur Google Scholar.")
    for titre, texte in GUIDE.items():
        with st.expander(titre, expanded=(titre == "Ta routine")):
            st.markdown(texte)


# ================================================================== NAVIGATION
NAV = {
    "Suivi": [
        st.Page(page_direct, title="Tableau de bord", icon=":material/dashboard:", default=True),
        st.Page(page_score, title="Score du concours", icon=":material/emoji_events:", url_path="score"),
        st.Page(page_lignes, title="Lignes", icon=":material/list_alt:", url_path="lignes"),
        st.Page(page_journal, title="Journal", icon=":material/receipt_long:", url_path="journal"),
    ],
    "Décider": [
        st.Page(page_candidats, title="Candidats", icon=":material/travel_explore:", url_path="candidats"),
        st.Page(page_simulateur, title="Simulateur", icon=":material/science:", url_path="simulateur"),
        st.Page(page_reequilibrage, title="Rééquilibrage", icon=":material/balance:", url_path="reequilibrage"),
        st.Page(page_regles, title="Règles", icon=":material/gavel:", url_path="regles"),
    ],
    "Risque": [
        st.Page(page_risque, title="Risque", icon=":material/shield:", url_path="risque"),
        st.Page(page_stress, title="Stress tests", icon=":material/thunderstorm:", url_path="stress-tests"),
        st.Page(page_regime, title="Régime de marché", icon=":material/waves:", url_path="regime"),
        st.Page(page_optimisation, title="Optimisation", icon=":material/tune:", url_path="optimisation"),
    ],
    "Analyse": [
        st.Page(page_performance, title="Performance", icon=":material/trending_up:", url_path="performance"),
        st.Page(page_momentum, title="Momentum", icon=":material/speed:", url_path="momentum"),
        st.Page(page_valorisation, title="Valorisation", icon=":material/sell:", url_path="valorisation"),
        st.Page(page_chance, title="Chance ou talent", icon=":material/casino:", url_path="chance-ou-talent"),
    ],
    "Outils": [
        st.Page(page_rapport, title="Rapport", icon=":material/summarize:", url_path="rapport"),
        st.Page(page_donnees, title="Données", icon=":material/database:", url_path="donnees"),
        st.Page(page_guide, title="Guide", icon=":material/menu_book:", url_path="guide"),
    ],
}
pg = st.navigation(NAV, expanded=True)
st.session_state["page"] = pg.title
bandeau()
pg.run()
