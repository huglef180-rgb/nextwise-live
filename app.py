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
try:
    import glossaire as GL
except Exception:   # si glossaire.py n'a pas été envoyé avec le reste, le site marche quand même, sans les « i »
    class GL:
        @staticmethod
        def bulle(_):
            return ""

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
    "Dissolution en France (juin 2024)": ("2024-06-07", "2024-06-28"),
}
EPISODES_POL = {
    "Dissolution de l'Assemblée : la première semaine (juin 2024)": ("2024-06-07", "2024-06-14"),
    "Dissolution : jusqu'à la veille du premier tour (juin 2024)": ("2024-06-07", "2024-06-28"),
    "Soulagement au lendemain du premier tour (avril 2017)": ("2017-04-21", "2017-04-24"),
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
    v = round(float(x) * 100, d)
    if v == 0:   # évite « -0 % » et « +0 % » sur une valeur arrondie à zéro
        return f"{0.0:.{d}f}".replace(".", ",") + " %"
    s = f"{v:+.{d}f}" if signe else f"{v:.{d}f}"
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
        # la nuit et le week-end, Yahoo renvoie une dernière ligne où seules les devises cotent ; plus quelques jours fériés locaux
        if df.shape[1]:
            df.loc[df.index[-1] + pd.Timedelta(days=1)] = [df[c].iloc[-1] if c.endswith("=X") else np.nan for c in df.columns]
        for k, c in enumerate(df.columns):
            if not c.endswith("=X"):
                df.iloc[[-23 - (k % 4), -254 + (k % 3), -400 - k], k] = np.nan
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
.nw-i {display:inline-flex; align-items:center; justify-content:center; width:14px; height:14px; margin-left:5px; border-radius:50%;
  border:1px solid rgba(128,128,128,.9); color:rgba(128,128,128,1); font:700 9px/1 Georgia, serif; text-transform:none; letter-spacing:0; cursor:help; vertical-align:middle;}
.nw-i:hover, .nw-i:focus {border-color:#4F8BFF; color:#4F8BFF; outline:none;}
.nw-tip {display:none; position:fixed; left:50%; bottom:20px; transform:translateX(-50%); width:min(640px, 92vw); z-index:999999; background:#111827; color:#f3f4f6;
  border-radius:12px; padding:14px 18px 16px; font:400 .86rem/1.55 'Inter', 'Segoe UI', system-ui, sans-serif; text-transform:none; letter-spacing:0; white-space:normal;
  text-align:left; box-shadow:0 12px 34px rgba(0,0,0,.4); cursor:default;}
.nw-i:hover .nw-tip, .nw-i:focus .nw-tip {display:block;}
.nw-tip .h {display:block; font-weight:700; font-size:.95rem; margin-bottom:2px;}
.nw-tip .l {display:block; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#93c5fd; margin-top:9px;}
.nwt {overflow-x:auto; margin:4px 0 0.8rem; border:1px solid rgba(128,128,128,.20); border-radius:12px}
.nwt table {border-collapse:collapse; font-size:0.85rem; width:100%; font-variant-numeric:tabular-nums}
.nwt thead th {text-align:left; padding:9px 10px; background:rgba(128,128,128,.07); border-bottom:1px solid rgba(128,128,128,.25);
  white-space:nowrap; font-weight:600; font-size:.7rem; text-transform:uppercase; letter-spacing:.03em}
.nwt td {padding:7px 10px; border-bottom:1px solid rgba(128,128,128,.12); white-space:nowrap}
.nwt tbody tr:last-child td {border-bottom:none}
.nwt tbody tr:hover td {background:rgba(79,139,255,.07)}
.nwt td:first-child {font-weight:500}
</style>"""


def tableau(data, column_config=None, hide_index=None, height=None, **_):
    """Tableau HTML statique : il s'affiche sans composant chargé à la demande par le navigateur,
    donc il ne peut pas « échouer ». Accepte les mêmes arguments que les appels st.dataframe d'origine."""
    sty = data if hasattr(data, "relabel_index") and hasattr(data, "to_html") else data.style.format(na_rep="")
    noms = []
    for c in sty.data.columns:
        lab = (column_config or {}).get(c, c)
        lab = lab if isinstance(lab, str) else str(c)
        noms.append(html.escape(lab) + GL.bulle(lab))
    sty = sty.relabel_index(noms, axis="columns")
    sty = sty.hide(axis="index")   # l'index n'est jamais affiché : une colonne qui compte doit être une vraie colonne
    # chiffres alignés à droite (colonnes numériques, en-tête compris), comme dans un tableur
    num = [i for i, c in enumerate(sty.data.columns) if pd.api.types.is_numeric_dtype(sty.data[c]) and not pd.api.types.is_bool_dtype(sty.data[c])]
    if num:
        sty = sty.set_table_styles([{"selector": f"th.col{i}, td.col{i}", "props": "text-align:right"} for i in num], overwrite=False)
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


FICHIER_RES = "/tmp/nextwise_resultats.json"


def _ligne_compta(df, noms):
    """Valeurs d'un poste comptable Yahoo, de l'exercice le plus récent au plus ancien."""
    if df is None or getattr(df, "empty", True):
        return []
    for n in noms:
        if n in df.index:
            v = pd.to_numeric(df.loc[n], errors="coerce")
            v.index = pd.to_datetime(v.index, errors="coerce")
            return v.dropna().sort_index(ascending=False).tolist()
    return []


def _ligne_periode(df, periodes=("0y", "+1y")):
    if df is None or getattr(df, "empty", True):
        return None
    for per in periodes:
        if per in df.index:
            return {str(k).lower(): v for k, v in df.loc[per].items()}
    return None


def _resultats_un(t, secteur=None):
    """Surprise de résultats, révisions des analystes et comptes annuels d'un titre. Chaque bloc est indépendant :
    une donnée absente chez Yahoo laisse un trou au lieu de tout bloquer."""
    vide = {"gp_a": np.nan, "croiss_actif": np.nan, "emission": np.nan, "surprise": np.nan, "date_pub": None,
            "n_pos4": np.nan, "n_4": np.nan, "rev30": np.nan, "rev90": np.nan, "net_rev30": np.nan,
            "croiss_ca": np.nan, "croiss_bpa": np.nan, "fcf_rdt": np.nan}
    d = dict(vide)
    if MOCK or yf is None:
        import zlib
        g = np.random.default_rng(zlib.crc32(t.encode()))
        fin = isinstance(secteur, str) and secteur.lower().startswith("financ")
        d.update({"gp_a": np.nan if fin else g.uniform(0.05, 0.55), "croiss_actif": g.normal(0.06, 0.08), "emission": g.normal(0.0, 0.02),
                  "surprise": g.normal(0.01, 0.07), "date_pub": (pd.Timestamp.now() - pd.Timedelta(days=int(g.integers(3, 120)))).strftime("%Y-%m-%d"),
                  "n_pos4": int(g.integers(0, 5)), "n_4": 4, "rev30": g.normal(0, 0.03), "rev90": g.normal(0, 0.06), "net_rev30": int(g.integers(-4, 5)),
                  "croiss_ca": g.normal(0.05, 0.08), "croiss_bpa": g.normal(0.07, 0.20), "fcf_rdt": np.nan if fin else g.normal(0.045, 0.03)})
        if os.environ.get("NEXTWISE_DEGRADE") == "1":
            for k in list(d):
                if g.random() < 0.35:
                    d[k] = None if k == "date_pub" else np.nan
        return d
    tk = yf.Ticker(t)
    try:
        inc, bal = tk.income_stmt, tk.balance_sheet
        gp = _ligne_compta(inc, ["Gross Profit"])
        ta = _ligne_compta(bal, ["Total Assets"])
        sh = _ligne_compta(bal, ["Ordinary Shares Number", "Share Issued"])
        d["gp_a"] = A.rentabilite_brute(gp[0], ta[0]) if gp and ta else np.nan
        d["croiss_actif"] = A.variation_annuelle(ta[0], ta[1]) if len(ta) > 1 else np.nan
        d["emission"] = A.variation_annuelle(sh[0], sh[1]) if len(sh) > 1 else np.nan
        ca = _ligne_compta(inc, ["Total Revenue", "Operating Revenue"])
        bpa = _ligne_compta(inc, ["Diluted EPS", "Basic EPS"])
        d["croiss_ca"] = A.variation_annuelle(ca[0], ca[1]) if len(ca) > 1 else np.nan
        d["croiss_bpa"] = A.revision_prevision(bpa[0], bpa[1]) if len(bpa) > 1 else np.nan      # en % de la valeur absolue : gère un BPA négatif
    except Exception as e:
        log(f"comptes {t} : {type(e).__name__}")
    try:
        fcf = _ligne_compta(tk.cashflow, ["Free Cash Flow"])
        cap = float(tk.fast_info["market_cap"])
        d["fcf_rdt"] = fcf[0] / cap if fcf and cap and cap > 0 else np.nan
    except Exception as e:
        log(f"free cash flow {t} : {type(e).__name__}")
    try:
        ed = tk.get_earnings_dates(limit=12)
        if ed is not None and not ed.empty:
            ed = ed.copy()
            idx = pd.DatetimeIndex(pd.to_datetime(ed.index, errors="coerce"))
            ed.index = idx.tz_convert(None) if idx.tz is not None else idx
            c_s = next((c for c in ed.columns if "surprise" in str(c).lower()), None)
            c_r = next((c for c in ed.columns if "reported" in str(c).lower()), None)
            if c_s and c_r:
                passe = ed[pd.to_numeric(ed[c_r], errors="coerce").notna()].sort_index(ascending=False)
                sur = pd.to_numeric(passe[c_s], errors="coerce")
                if len(passe) and pd.notna(sur.iloc[0]):
                    d["surprise"] = float(sur.iloc[0]) / 100
                    d["date_pub"] = passe.index[0].strftime("%Y-%m-%d")
                d["n_4"] = int(sur.iloc[:4].notna().sum())
                d["n_pos4"] = int((sur.iloc[:4] > 0).sum())
    except Exception as e:
        log(f"surprises {t} : {type(e).__name__}")
    try:
        tr = _ligne_periode(tk.eps_trend)
        if tr:
            d["rev30"] = A.revision_prevision(tr.get("current"), tr.get("30daysago"))
            d["rev90"] = A.revision_prevision(tr.get("current"), tr.get("90daysago"))
    except Exception as e:
        log(f"tendance BPA {t} : {type(e).__name__}")
    try:
        rv = _ligne_periode(tk.eps_revisions)
        if rv:
            up, dn = pd.to_numeric(rv.get("uplast30days"), errors="coerce"), pd.to_numeric(rv.get("downlast30days"), errors="coerce")
            d["net_rev30"] = (0 if pd.isna(up) else up) - (0 if pd.isna(dn) else dn) if pd.notna(up) or pd.notna(dn) else np.nan
    except Exception as e:
        log(f"révisions {t} : {type(e).__name__}")
    return d


@st.cache_data(ttl=24 * 3600, show_spinner="Résultats, révisions et comptes annuels (Yahoo, une à deux minutes la première fois)…")
def donnees_resultats(tickers, secteurs=None):
    import json
    sect = dict(secteurs or ())
    out = {t: _resultats_un(t, sect.get(t)) for t in tickers}
    try:
        ancien = {}
        if os.path.exists(FICHIER_RES):
            with open(FICHIER_RES) as f:
                ancien = json.load(f).get("donnees", {})
        ancien.update({t: {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else v) for k, v in d.items()} for t, d in out.items()})
        with open(FICHIER_RES, "w") as f:
            json.dump({"maj": dt.datetime.now().isoformat(), "donnees": ancien}, f)
    except Exception as e:
        log(f"fichier résultats : {type(e).__name__}")
    return pd.DataFrame(out).T


def lire_resultats():
    """Dernières données de résultats enregistrées sur le serveur (moins de 24 h), sans appel à Yahoo."""
    import json
    try:
        with open(FICHIER_RES) as f:
            j = json.load(f)
        if (dt.datetime.now() - dt.datetime.fromisoformat(j["maj"])).total_seconds() < 24 * 3600:
            return pd.DataFrame(j["donnees"]).T, j["maj"]
    except Exception:
        pass
    return pd.DataFrame(), None


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
    st.number_input("Taux sans risque (%)", 0.0, 10.0, 2.5, 0.25, key="rf",
                    help="Taux de la facilité de dépôt de la BCE (2,50 % depuis septembre 2026), utilisé pour les ratios de Sharpe, Sortino, Treynor et M². Vérifie-le sur le site de la BCE.")
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
# stop suiveur proportionnel à la volatilité de chaque titre, depuis le plus haut atteint depuis le premier achat
stops = {}
try:
    premiers = tx[tx["type"].astype(str).str.lower() == "achat"].groupby("ticker")["date"].min()
    for t, r in pf.iterrows():
        if t in adj_eur.columns and not r.get("cours_absent", False):
            sp = A.stop_suiveur(adj_eur[t], premiers.get(t), r.get("vol_ewma") if pd.notna(r.get("vol_ewma")) else r.get("vol"))
            if sp:
                stops[t] = sp
                if sp["touche"]:
                    alertes.append(("bad", f"{r['nom']} : {pct(sp['recul'], 0, True)} depuis son plus haut du {sp['date_haut']:%d/%m}, au-delà de son stop suiveur "
                                           f"({pct(sp['seuil'], 0, True)}, soit 2,5 fois sa volatilité mensuelle). Relis sa thèse : si elle ne tient plus, sors."))
except Exception as e:
    log(f"stop suiveur : {type(e).__name__}: {e}")
cal_dates, cal_maj = lire_calendrier()
if cal_dates:
    auj_ = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    for t, d in cal_dates.items():
        if d and t in pf.index:
            jj = (pd.Timestamp(d) - auj_).days
            if 0 <= jj <= 3:
                alertes.append(("warn", f"{pf.loc[t, 'nom']} publie ses résultats {'aujourd’hui' if jj == 0 else f'dans {jj} jour' + ('s' if jj > 1 else '')} ({pd.Timestamp(d):%d/%m}) : relis sa condition de sortie avant."))
res_alerte, _ = lire_resultats()
if not res_alerte.empty:
    auj_r = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    for t in pf.index:
        if t not in res_alerte.index:
            continue
        rr_ = res_alerte.loc[t]
        sg, jj = A.signal_pead(pd.to_numeric(rr_.get("surprise"), errors="coerce"), rr_.get("date_pub"), auj_r)
        if sg == -1:
            alertes.append(("warn", f"{pf.loc[t, 'nom']} : résultats {pct(abs(float(rr_['surprise'])), 0)} sous le consensus il y a {jj} jours. "
                                    "La dérive post-annonce joue contre la ligne pendant environ trois mois."))
        r90 = pd.to_numeric(rr_.get("rev90"), errors="coerce")
        if pd.notna(r90) and r90 < -0.05:
            alertes.append(("warn", f"{pf.loc[t, 'nom']} : prévisions de bénéfice abaissées de {pct(-r90, 0)} en 90 jours par les analystes."))
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
.nw-k .t {font-size:.7rem; text-transform:uppercase; letter-spacing:.07em;}
.nw-k .t .tt {opacity:.65;}
.nw-i {display:inline-flex; align-items:center; justify-content:center; width:14px; height:14px; margin-left:5px; border-radius:50%;
  border:1px solid rgba(128,128,128,.9); color:rgba(128,128,128,1); font:700 9px/1 Georgia, serif; text-transform:none; letter-spacing:0; cursor:help; vertical-align:middle;}
.nw-i:hover, .nw-i:focus {border-color:#4F8BFF; color:#4F8BFF; outline:none;}
.nw-tip {display:none; position:fixed; left:50%; bottom:20px; transform:translateX(-50%); width:min(640px, 92vw); z-index:999999; background:#111827; color:#f3f4f6;
  border-radius:12px; padding:14px 18px 16px; font:400 .86rem/1.55 'Inter', 'Segoe UI', system-ui, sans-serif; text-transform:none; letter-spacing:0; white-space:normal;
  text-align:left; box-shadow:0 12px 34px rgba(0,0,0,.4); cursor:default;}
.nw-i:hover .nw-tip, .nw-i:focus .nw-tip {display:block;}
.nw-tip .h {display:block; font-weight:700; font-size:.95rem; margin-bottom:2px;}
.nw-tip .l {display:block; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#93c5fd; margin-top:9px;}
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
.nw-intro {font-size:.9rem; opacity:.8; margin:-4px 0 12px; max-width: 1100px; line-height:1.5;}
.nw-k {transition: border-color .15s;} .nw-k:hover {border-color: rgba(79,139,255,.45);}
.nw-band .nw-k:first-child {border-color: rgba(79,139,255,.40); background: rgba(79,139,255,.06);}
.nw-al {display:flex; flex-direction:column; gap:6px; margin:2px 0 12px;}
.nw-al .a {display:flex; gap:10px; align-items:flex-start; padding:8px 12px; border-radius:10px; font-size:.86rem; line-height:1.4;
  background:rgba(127,127,127,.05); border:1px solid rgba(127,127,127,.15);}
.nw-al .a.bad {border-left:3px solid #ef4444;} .nw-al .a.warn {border-left:3px solid #f59e0b;}
.nw-al .tag {flex:none; font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; padding:2px 7px; border-radius:6px; margin-top:1px;}
.nw-al .bad .tag {background:rgba(239,68,68,.14); color:#dc2626;} .nw-al .warn .tag {background:rgba(245,158,11,.16); color:#b45309;}
.nw-al .ok {border-left:3px solid #22c55e;}
[data-testid="stNavSectionHeader"] {font-size:.68rem !important; text-transform:uppercase; letter-spacing:.09em; opacity:.6; font-weight:650 !important; margin-top:.4rem;}
[data-testid="stSidebarNavLink"] {border-radius:8px;}
[data-testid="stMainBlockContainer"] h3 {padding-top:.2rem;}
[data-testid="stExpander"] details {border-radius:12px;}
div[data-testid="stAlert"] {border-radius:12px;}
@media print {
  html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"]
    {height:auto !important; max-height:none !important; overflow:visible !important; position:static !important;}
  [data-testid="stSidebar"], header, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
  .js-plotly-plot, .nwt, .nw-k {break-inside:avoid;}
}
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
           "".join(f'<div class="nw-k"><div class="t"><span class="tt">{t}</span>{GL.bulle(t)}</div><div class="v">{v}</div><div class="s">{s_}</div></div>' for t, v, s_ in cartes) + "</div>"
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
    conteneur.html(f'<div class="nw-k solo"><div class="t"><span class="tt">{html.escape(str(titre))}</span>{GL.bulle(titre)}</div><div class="v">{v}</div>{s_}</div>')


def RF():
    return float(st.session_state.get("rf", 2.5)) / 100


def intro(texte):
    st.html(f'<div class="nw-intro">{texte}</div>')


def liste_alertes(liste, maxi=6):
    """Alertes en liste compacte : rouge = règle touchée, orange = à surveiller. Les suivantes sont repliées."""
    if not liste:
        st.html(CSS_GLOBAL + '<div class="nw-al"><div class="a ok">Rien à signaler.</div></div>')
        return
    tri = sorted(liste, key=lambda a: a[0] != "bad")
    ligne = lambda n, t: f'<div class="a {n}"><span class="tag">{"Alerte" if n == "bad" else "À voir"}</span><span>{html.escape(t)}</span></div>'
    st.html(CSS_GLOBAL + '<div class="nw-al">' + "".join(ligne(n, t) for n, t in tri[:maxi]) + "</div>")
    if len(tri) > maxi:
        with st.expander(f"{len(tri) - maxi} autre{'s' if len(tri) - maxi > 1 else ''}"):
            st.html(CSS_GLOBAL + '<div class="nw-al">' + "".join(ligne(n, t) for n, t in tri[maxi:]) + "</div>")


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
        liste_alertes(alertes)
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
    borne = max(float(tri["var_jour"].abs().max()) * 1.35, 0.002) if len(tri) else 0.01   # marge pour les étiquettes, des deux côtés
    fig.update_layout(height=max(300, 22 * len(tri)), margin=dict(t=10, l=0, r=40, b=0), xaxis_tickformat="+.1%", xaxis_range=[-borne, borne])
    graphe(fig, key="varjour")

    st.subheader("Par où passer")
    st.caption("Le site compte beaucoup de pages : voici l'ordre dans lequel les ouvrir selon ce que tu viens faire.")
    for col, (titre, etapes) in zip(st.columns(len(PARCOURS)), PARCOURS.items()):
        with col:
            st.markdown(f"**{titre}**")
            for i, (u, quoi) in enumerate(etapes, 1):
                if u in PAGES:
                    st.page_link(PAGES[u], label=f"{i}\\. {PAGES[u].title}")
                    st.caption(quoi)

# ================================================================== LIGNES
def page_lignes():
    st.subheader("Toutes les lignes")
    st.caption("Le tableau donne l'état de chaque position. Plus bas, la fiche d'une ligne rassemble sa thèse, sa menace, sa condition de sortie et son stop : "
               "c'est la page à ouvrir quand une alerte cite un titre.")
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
        if sel in stops:
            sp_ = stops[sel]
            st.caption(f"Stop suiveur : {pct(sp_['recul'], 0, True)} depuis le plus haut du {sp_['date_haut']:%d/%m/%Y}, pour un seuil à {pct(sp_['seuil'], 0, True)} "
                       f"(2,5 fois la volatilité mensuelle du titre){' : seuil franchi' if sp_['touche'] else ''}.")
        res_l, _ = lire_resultats()
        if not res_l.empty and sel in res_l.index:
            q = res_l.loc[sel]
            num = lambda k: pd.to_numeric(q.get(k), errors="coerce")
            sg, jj = A.signal_pead(num("surprise"), q.get("date_pub"), pd.Timestamp.now(tz=TZ).normalize().tz_localize(None))
            n1, n2, n3 = st.columns(3)
            kpi(n1, "Dernière surprise", pct(num("surprise"), 1, True) if pd.notna(num("surprise")) else "—", _derive_txt(sg, jj))
            kpi(n2, "Révision BPA 90 j", pct(num("rev90"), 1, True) if pd.notna(num("rev90")) else "—")
            kpi(n3, "Rentabilité brute", pct(num("gp_a"), 0) if pd.notna(num("gp_a")) else "—",
                f"actif {pct(num('croiss_actif'), 0, True)} sur un an" if pd.notna(num("croiss_actif")) else None, delta_color="inverse")
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
    intro("Les indicateurs des logiciels de gestion (Morningstar Direct, Bloomberg PORT), pour ton portefeuille et pour la référence choisie, sur la même période. "
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
    refs = ["Euronext 100", "Ta liste de départ à poids égaux"]
    ref = st.radio("Comparer à", refs, horizontal=True, key="ratios_ref",
                   help="Un portefeuille à poids égaux bat souvent l'indice pondéré par la capitalisation, sans aucun talent de sélection "
                        "(DeMiguel, Garlappi et Uppal, 2009). Battre aussi ta liste à poids égaux montre que l'avance vient du choix des titres, pas de la pondération.")
    if ref == refs[1]:
        pu_, _, _ = prix_univers(True)
        r_ind = pu_.pct_change(fill_method=None).mean(axis=1, skipna=True) if not pu_.empty else None
    else:
        r_ind = adj_eur[BENCH_PRINCIPAL].pct_change(fill_method=None) if BENCH_PRINCIPAL in adj_eur else None
    nom_ref = "Euronext 100" if ref == refs[0] else "Liste à poids égaux"
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
                hide_index=True, column_config={"ratio": "Indicateur", "portefeuille_txt": "Portefeuille", "indice_txt": nom_ref, "verdict": "Face à la référence",
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
    intro("Combien de jours il faudrait pour vendre chaque ligne sans peser sur le cours, en ne prenant pas plus de 10 % du volume quotidien moyen "
          "(la règle prudente des gérants : 5 à 10 % du volume d'une séance). Le règlement de NextWise précise que l'exécution dépend de la liquidité : une ligne peu échangée "
          "peut être exécutée à un moins bon prix.")
    vm = volumes_moyens(tuple(pf.index))
    if vm is None or vm.empty:
        st.info("Volumes indisponibles pour l'instant.")
    else:
        lq = pd.DataFrame({"nom": pf["nom"], "quantite": pf["quantite"], "volume": vm.reindex(pf.index)})
        lq["part"] = lq["quantite"] / lq["volume"]
        lq["jours"] = lq["quantite"] / (0.1 * lq["volume"])
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
    if MOCK:   # test hors ligne : facteurs simulés (publiés avec un mois de retard, comme les vrais) ; vides en mode dégradé
        if os.environ.get("NEXTWISE_DEGRADE") == "1":
            return pd.DataFrame()
        fin = (pd.Timestamp.now(tz=TZ).normalize().tz_localize(None) - pd.offsets.MonthEnd(1))
        idx = pd.bdate_range(end=fin, periods=1900)
        rng = np.random.default_rng(11)
        return pd.DataFrame({c: rng.normal(m, s, len(idx)) for c, m, s in [
            ("Taille (SMB)", 0.0, 0.004), ("Value (HML)", 0.0001, 0.005), ("Rentabilité (RMW)", 0.0001, 0.003),
            ("Investissement (CMA)", 0.0, 0.003), ("Momentum (WML)", 0.0003, 0.007)]}, index=idx)
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
            cum_k = (1 + wml_k).cumprod()
            recul_k = float(cum_k.iloc[-1] / cum_k.max() - 1)
            st.markdown("### Faut-il se fier au momentum en ce moment ?")
            c = st.columns(4)
            ech = kr_k["echelle_bsc"]
            kpi(c[3], "Recul du facteur depuis son plus haut", pct(recul_k, 0, True),
                f"plus haut le {cum_k.idxmax():%d/%m/%Y}" if recul_k < -0.005 else "au plus haut", delta_color="off")
            kpi(c[0], "Poids à donner au momentum", "—" if pd.isna(ech) else f"{ech:.1f}×".replace(".", ","),
                "1 = normal · sous 1 = moins de poids (Barroso et Santa-Clara, 2015)", delta_color="off")
            kpi(c[1], "Volatilité du facteur momentum", pct(kr_k["vol_wml"], 0), "6 derniers mois · cible 12 %", delta_color="off")
            kpi(c[2], "Marché sur 2 ans", pct(kr_k["marche_2ans"], 0, True),
                "baissier : terrain des krachs du momentum" if kr_k["marche_baissier"] else "pas baissier (Daniel et Moskowitz, 2016)", delta_color="off")
            if kr_k["alerte"]:
                st.error("Risque de krach du momentum : marché baissier sur 2 ans et volatilité forte. Ne t'appuie pas sur le momentum pour tes prochains choix.")
            elif pd.notna(ech) and ech < 0.8:
                st.warning("Le momentum est agité en ce moment" + (f" (le facteur est à {pct(recul_k, 0, True)} de son plus haut)" if recul_k < -0.10 else "")
                           + " : donne-lui moins de poids que d'habitude. Ne vends pas une ligne uniquement pour un score de "
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
                ff_m = french_europe()
                off = None
                if ff_m is not None and not ff_m.empty and "Momentum (WML)" in ff_m:
                    o = ff_m["Momentum (WML)"].dropna()
                    o = o[o.index > wml.index.min()]
                    if len(o) > 60:
                        off = (1 + o).cumprod() - 1
                g, d = st.columns(2)
                with g:
                    fig = go.Figure(go.Scatter(x=cumul.index, y=cumul, line=dict(width=2), name=f"Ta liste de départ ({pu.shape[1]} titres)"))
                    if off is not None:
                        fig.add_trace(go.Scatter(x=off.index, y=off, line=dict(width=2, dash="dot", color="#64748B"),
                                                 name="Europe entière (Kenneth French)"))
                    fig.add_hline(y=0, line_color="#bbbbbb", line_width=1)
                    fig.update_layout(height=300, yaxis_tickformat="+.0%", margin=dict(t=30, l=0, r=0, b=0), title=dict(text="Rendement cumulé du facteur", font_size=13),
                                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
                    graphe(fig, key="wml")
                with d:
                    fig = go.Figure(go.Scatter(x=vol6.index, y=vol6, line=dict(width=2, color=C_ATTENTION)))
                    fig.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(t=30, l=0, r=0, b=0), title=dict(text="Volatilité du facteur sur 6 mois", font_size=13))
                    graphe(fig, key="wmlvol")
                kr = A.risque_krach_momentum(adj_eur[BENCH_PRINCIPAL], wml)
                c = st.columns(4)
                kpi(c[0], "Marché sur 2 ans", pct(kr["marche_2ans"], 0, True), "baissier : danger" if kr["marche_baissier"] else "pas baissier")
                kpi(c[1], "Vol. du marché", pct(kr["vol_marche"], 0), f"6 mois · alerte au-delà de {pct(kr['seuil_vol'], 0)}")
                kpi(c[2], "Vol. du momentum", pct(kr["vol_wml"], 0), "6 derniers mois")
                kpi(c[3], "Exposition conseillée", "—" if pd.isna(kr["echelle_bsc"]) else f"{kr['echelle_bsc']:.1f}×".replace(".", ","),
                            "Barroso et Santa-Clara (2015)", delta_color="off")
                cum1 = 1 + cumul
                recul = float(cum1.iloc[-1] / cum1.max() - 1)
                c = st.columns(4)
                kpi(c[0], f"Ta liste depuis {cumul.index.min():%m/%Y}", pct(cumul.iloc[-1], 0, True), "gagnants − perdants")
                if off is not None:
                    kpi(c[1], "Europe entière", pct(off.iloc[-1], 0, True), f"même période · jusqu'au {off.index.max():%d/%m/%Y}")
                else:
                    kpi(c[1], "Europe entière", "—", "données Kenneth French indisponibles", delta_color="off")
                kpi(c[2], "Recul depuis le plus haut", pct(recul, 0, True), f"plus haut le {cum1.idxmax():%d/%m/%Y}" if recul < -0.005 else "au plus haut", delta_color="off")
                an = wml.groupby(wml.index.year).agg(lambda x: (1 + x).prod() - 1 if len(x) >= 120 else np.nan).dropna()
                kpi(c[3], "Meilleure / pire année",
                    f"{pct(an.max(), 0, True)} / {pct(an.min(), 0, True)}" if len(an) else "—",
                    f"{an.idxmax()} / {an.idxmin()}" if len(an) else "", delta_color="off")
                if kr["alerte"]:
                    st.error("Risque de krach du momentum : marché en baisse sur 2 ans et volatilité forte. C'est dans ce contexte que le momentum s'effondre, "
                             "quand les perdants rebondissent violemment (Daniel et Moskowitz, 2016). Ne t'appuie pas sur le momentum pour tes prochains achats.")
                elif pd.notna(kr["echelle_bsc"]) and kr["echelle_bsc"] < 0.8:
                    st.warning("Pas de krach au sens de Daniel et Moskowitz (le marché n'est pas baissier sur 2 ans), mais le momentum est agité : sa volatilité "
                               f"({pct(kr['vol_wml'], 0)}) dépasse nettement la cible de 12 %"
                               + (f" et le facteur est à {pct(recul, 0, True)} de son plus haut" if recul < -0.10 else "")
                               + ". Donne-lui moins de poids que d'habitude dans tes choix.")
                else:
                    st.success("Pas de signal de krach et volatilité normale : le contexte est normal pour utiliser le momentum.")
                st.markdown(
                    "**Comment lire ce graphique sans se tromper**\n"
                    "- **Ce n'est pas ta stratégie.** Ce facteur achète les gagnants *et vend à découvert* les perdants. Toi, tu ne vends rien à découvert. "
                    "Or les krachs du momentum viennent surtout de la jambe vendeuse : ce sont les perdants qui rebondissent violemment (Daniel et Moskowitz, 2016).\n"
                    f"- **Environ {max(1, pu.shape[1] // 3)} titres de chaque côté, c'est très peu.** Le facteur officiel utilise des centaines d'actions européennes. Sur ta liste, quelques "
                    "titres suffisent à faire la courbe : compare avec la courbe « Europe entière ». Si elle monte quand la tienne baisse, c'est ton univers qui est en cause, pas le momentum.\n"
                    "- **Quelques années ne prouvent rien.** Avec un ratio de Sharpe d'environ 0,5, l'ordre de grandeur historique du momentum, il faut environ "
                    "(2 ÷ 0,5)² = 16 ans pour qu'une prime soit significative, et une période de 5 à 6 ans perdante arrive environ une fois sur dix. C'est le même raisonnement que la page Chance ou talent.\n"
                    "- **Biais de sélection.** Ta liste a été choisie en 2026. Des titres qui étaient perdants il y a quelques années et sont devenus des vedettes "
                    "se retrouvent dans la jambe vendeuse au début de la période, ce qui pénalise artificiellement le facteur.\n"
                    "- **Ce qui compte pour tes décisions**, ce sont la volatilité récente du facteur et son recul depuis le plus haut : quand ils montent, "
                    "le momentum doit peser moins dans tes choix (Barroso et Santa-Clara, 2015).")
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
    st.caption("Tes règles d'investissement vérifiées automatiquement, ligne par ligne puis pour le portefeuille entier. "
               "C'est la preuve, pour le jury, que tu suis un processus et non ton humeur.")
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
    if source == "Google Sheet":
        st.caption("Ajoute chaque ordre exécuté dans l'onglet « transactions » de ton Google Sheet : le site le prend en compte dans la minute.")
    else:
        st.caption("Après chaque ordre, ajoute une ligne à transactions.csv sur GitHub (icône crayon) : date AAAA-MM-JJ, ticker Yahoo, Achat ou Vente, quantité, "
                   "prix en euros avec un point décimal. Le site se met à jour une à deux minutes après l'enregistrement.")
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
    st.caption("À ouvrir quand un chiffre te paraît bizarre : cette page dit pour chaque titre si les cours sont à jour, complets et sans saut suspect. "
               "Un titre en défaut fausse le risque et la performance des autres pages.")
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
# ================================================================== SCÉNARIOS POLITIQUES
@st.cache_data(ttl=3600, show_spinner="Indices français et européens…")
def indices_politiques():
    """CAC 40 et un indice européen large (le premier disponible) pour construire le facteur « France moins Europe »."""
    eu = {"^STOXX": "STOXX Europe 600", "^STOXX50E": "Euro Stoxx 50", "^GDAXI": "DAX"}
    cl, _ = historique(("^FCHI",) + tuple(eu))
    if cl is None or cl.empty or "^FCHI" not in cl.columns or cl["^FCHI"].dropna().shape[0] < 300:
        return None
    for t, nom in eu.items():
        if t in cl.columns and cl[t].dropna().shape[0] > 300:
            return {"fr": cl["^FCHI"].dropna(), "eu": cl[t].dropna(), "nom_eu": nom}
    return None


@st.cache_data(ttl=900, show_spinner="Cotes du marché de prédiction…")
def cotes_prediction(slug):
    """Probabilités implicites d'un marché de prédiction (Polymarket) : dernière cote de chaque issue et historique quotidien
    des principales. Renvoie None si la source ne répond pas : la page passe alors en saisie manuelle."""
    import json, urllib.request
    if MOCK:
        if os.environ.get("NEXTWISE_DEGRADE") == "1":
            return None
        g = np.random.default_rng(21)
        idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=300)
        noms = [f"Candidat {c}" for c in "ABCDE"]
        h = pd.DataFrame({n: np.clip(p0 + np.cumsum(g.normal(0, 0.008, len(idx))), 0.01, 0.9) for n, p0 in zip(noms, [0.34, 0.27, 0.16, 0.09, 0.05])}, index=idx)
        return {"titre": "Marché simulé (mode test)", "actuel": h.iloc[-1].sort_values(ascending=False), "historique": h, "maj": dt.datetime.now()}
    def _json(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode("utf-8"))
    def _liste(v):
        return json.loads(v) if isinstance(v, str) else (v or [])
    try:
        ev = _json(f"https://gamma-api.polymarket.com/events?slug={slug}")
        ev = ev[0] if isinstance(ev, list) and ev else ev
        actuel, jetons = {}, {}
        for m in ev.get("markets", []):
            if m.get("closed") or m.get("archived"):
                continue
            prix, jet = _liste(m.get("outcomePrices")), _liste(m.get("clobTokenIds"))
            nom = str(m.get("groupItemTitle") or m.get("question") or "").strip()
            if nom and prix:
                actuel[nom] = float(prix[0])
                if jet:
                    jetons[nom] = jet[0]
        if not actuel:
            return None
        actuel = pd.Series(actuel).sort_values(ascending=False)
        hist = {}
        for nom in actuel.index[:8]:
            try:
                j = _json(f"https://clob.polymarket.com/prices-history?market={jetons[nom]}&interval=max&fidelity=1440")
                pts = j.get("history", [])
                if pts:
                    sr = pd.Series({pd.Timestamp(x["t"], unit="s").normalize(): float(x["p"]) for x in pts})
                    hist[nom] = sr[~sr.index.duplicated(keep="last")]
            except Exception as e:
                log(f"historique des cotes {nom} : {type(e).__name__}")
                break   # si l'historique ne répond pas, inutile d'attendre huit fois le délai
        return {"titre": str(ev.get("title", slug)), "actuel": actuel, "historique": pd.DataFrame(hist).sort_index(), "maj": dt.datetime.now()}
    except Exception as e:
        log(f"cotes du marché de prédiction : {type(e).__name__}: {e}")
        return None


def page_politique():
    st.subheader("Scénarios politiques : ce qu'une élection peut faire à ton portefeuille")
    intro("Cette page relie trois choses : les <b>probabilités</b> que le marché donne à chaque issue, la <b>sensibilité</b> de tes lignes au risque politique "
          "français, et ce que des <b>épisodes réels</b> ont fait à ces mêmes titres. Elle ne juge aucun programme : elle mesure seulement comment les marchés ont réagi, "
          "dans le passé, quand le risque politique montait ou retombait.")
    ind = indices_politiques()
    if ind is None:
        st.info("Les indices français et européens ne sont pas disponibles pour l'instant : réessaie dans quelques minutes.")
        return
    r_fr, r_eu = ind["fr"].pct_change(fill_method=None), ind["eu"].pct_change(fill_method=None)
    fac = (r_fr - r_eu).dropna()
    R = adj_eur[[t for t in pf.index if t in adj_eur.columns]].pct_change(fill_method=None)
    investi = float(poids.sum())
    total = float(pf["valeur"].sum() / investi) if investi > 0 else np.nan

    # ---------- 1. les cotes
    st.markdown("### 1. Les probabilités")
    c1, c2 = st.columns([1, 2])
    src_pol = c1.radio("Source", ["Marché de prédiction", "Saisie à la main"], key="pol_src",
                      help="Un marché de prédiction est une bourse où l'on parie sur une issue : le prix d'un contrat se lit comme une probabilité. "
                           "Il réagit plus vite que les sondages, mais il peut être peu liquide et déplacé par quelques gros parieurs.")
    cotes, p, serie_p = None, None, None
    if src_pol == "Marché de prédiction":
        slug = c2.text_input("Identifiant du marché sur Polymarket", "next-french-presidential-election", key="pol_slug",
                             help="C'est la fin de l'adresse de la page du marché sur polymarket.com/event/…")
        cotes = cotes_prediction(slug.strip())
        if cotes is None:
            st.warning("Les cotes ne sont pas accessibles pour l'instant (source indisponible ou identifiant introuvable). Passe en saisie à la main : le reste de la page fonctionne pareil.")
    if cotes is not None:
        act = cotes["actuel"].head(10)
        g, d = st.columns([3, 2])
        with g:
            fig = go.Figure(go.Bar(x=act.values[::-1], y=act.index[::-1], orientation="h", marker_color=PALETTE[0],
                                   text=[pct(v, 0) for v in act.values[::-1]], textposition="outside", hovertemplate="%{y} : %{x:.1%}<extra></extra>"))
            fig.update_layout(height=max(220, 30 * len(act)), margin=dict(t=36, l=0, r=30, b=0), xaxis_tickformat=".0%", xaxis_range=[0, max(0.1, float(act.max()) * 1.25)],
                              title=dict(text=f"{cotes['titre']} : probabilité implicite de chaque issue", font_size=13))
            graphe(fig, key="pol_cotes")
        with d:
            choisis = st.multiselect("Quelles issues inquiéteraient les marchés ?", list(cotes["actuel"].index), key="pol_tension",
                                     help="C'est toi qui définis le « scénario de tension ». En juin 2024, les marchés ont surtout réagi à l'incertitude budgétaire, "
                                          "quel que soit le camp. Le site additionne les probabilités des issues choisies.")
            if choisis:
                p = float(min(1.0, cotes["actuel"].reindex(choisis).sum()))
                h = cotes["historique"]
                cols = [c for c in choisis if c in h.columns]
                serie_p = h[cols].sum(axis=1, min_count=1).clip(upper=1.0).dropna() if cols else None
                if serie_p is not None and len(serie_p):
                    # cotes 7 jours sur 7, bourse 5 jours sur 7 : on garde la dernière cote connue à chaque séance
                    seances = fac.index[fac.index >= serie_p.index[0]]
                    serie_p = serie_p.reindex(serie_p.index.union(seances)).ffill().reindex(seances).dropna()
                kpi(st, "Probabilité du scénario de tension", pct(p, 0), f"mise à jour {cotes['maj']:%d/%m à %H:%M}", delta_color="off")
            else:
                st.caption("Choisis au moins une issue pour calculer la probabilité du scénario de tension. Les probabilités ne somment pas forcément à 100 % : "
                           "chaque issue est un contrat séparé.")
    if p is None:
        p = st.slider("Probabilité du scénario de tension (%)", 0, 100, 30, 5, key="pol_p",
                      help="À régler toi-même à partir des sondages ou des cotes que tu consultes.") / 100

    # ---------- 2. le marché réagit-il aux cotes ?
    reac = None
    if serie_p is not None and len(serie_p) > 40:
        st.markdown("### 2. Les cours réagissent-ils vraiment à ces probabilités ?")
        intro(f"Le <b>facteur France</b> est l'écart quotidien entre le CAC 40 et le {ind['nom_eu']}. S'il baisse les jours où la probabilité monte, "
              "c'est que la politique explique une partie des mouvements. Sinon, les baisses de tes titres viennent d'ailleurs.")
        fz = fac.reindex(serie_p.index).fillna(0.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie_p.index, y=serie_p, name="Probabilité du scénario de tension", line=dict(width=2, color=PALETTE[3])))
        fig.add_trace(go.Scatter(x=fz.index, y=(1 + fz).cumprod() - 1, name="Facteur France (cumulé)", yaxis="y2", line=dict(width=2, color=PALETTE[0])))
        fig.update_layout(height=320, margin=dict(t=10, l=0, r=0, b=0), yaxis=dict(tickformat=".0%", title="probabilité"),
                          yaxis2=dict(tickformat="+.0%", overlaying="y", side="right", showgrid=False, title="CAC 40 moins Europe"),
                          legend=dict(orientation="h", y=-0.15), hovermode="x unified")
        graphe(fig, key="pol_reac")
        reac = A.reaction_aux_cotes(fac, serie_p)
        c = st.columns(4)
        if reac:
            kpi(c[0], "Écart implicite entre les deux issues", pct(reac["pente"], 1, True), "pente de la régression", delta_color="off")
            kpi(c[1], "Statistique t", f"{reac['t']:.1f}".replace(".", ","), "solide au-delà de 2,6 en valeur absolue", delta_color="off")
            kpi(c[2], "Part des mouvements expliquée", pct(reac["r2"], 1), f"R² sur {reac['n']} séances", delta_color="off")
            dp = float(serie_p.iloc[-1] - serie_p.iloc[-2]) if len(serie_p) > 1 else 0.0
            kpi(c[3], "Dernière variation de la probabilité", f"{dp * 100:+.1f} pt".replace(".", ","), f"le {serie_p.index[-1]:%d/%m}", delta_color="off")
            if abs(reac["t"]) < 2.6:
                st.info("Pas de lien mesurable pour l'instant : les variations de la probabilité n'expliquent pas l'écart entre le CAC 40 et l'Europe. "
                        "Si tes titres français baissent, la cause est probablement ailleurs (résultats, taux, secteur). Le lien apparaît en général à l'approche du vote. "
                        "Le seuil est volontairement exigeant (2,6 et non 2) : en essayant plusieurs combinaisons d'issues, on finit toujours par trouver un lien dû au hasard.")
            elif reac["pente"] < 0:
                st.warning(f"Lien mesurable : quand la probabilité monte de 10 points, le CAC 40 perd environ {pct(-reac['pente'] * 0.10, 1)} face à l'Europe.")
            else:
                st.info("Le lien mesuré est positif : les marchés ne traitent pas ces issues comme un scénario de tension. Revois ta sélection d'issues.")
        else:
            st.caption("Pas encore assez de séances avec des cotes pour mesurer un lien.")

    # ---------- 3. sensibilités
    st.markdown("### 3. Quelles lignes portent le risque France ?")
    intro("Pour chaque ligne, une régression sur deux ans sépare ce qui vient du marché européen et ce qui vient du <b>facteur France</b>. "
          "Une sensibilité de 1 signifie : quand la France fait 1 point de moins que l'Europe, la ligne perd 1 point de plus que ce que son bêta laissait prévoir. "
          "Une action cotée à Paris n'est pas forcément exposée ; une banque italienne peut l'être.")
    sens = A.sensibilite_facteur(R, r_eu, fac)
    if sens.empty:
        st.info("Historique insuffisant pour estimer les sensibilités.")
        return
    sens["nom"] = pf["nom"].reindex(sens.index)
    sens["poids"] = poids.reindex(sens.index).fillna(0.0)
    sens["contrib"] = sens["poids"] * sens["beta_facteur"]
    beta_ptf = float(sens["contrib"].sum())
    z = sens.sort_values("contrib")
    g, d = st.columns([3, 2])
    with g:
        fig = go.Figure(go.Bar(x=z["contrib"], y=z["nom"], orientation="h", marker_color=[C_MAUVAIS if v > 0 else C_BON for v in z["contrib"]],
                               hovertemplate="%{y} : %{x:+.3f}<extra></extra>"))
        fig.update_layout(height=max(300, 20 * len(z)), margin=dict(t=36, l=0, r=10, b=0),
                          title=dict(text="Contribution de chaque ligne à la sensibilité du portefeuille (poids × sensibilité)", font_size=13))
        graphe(fig, key="pol_sens")
    with d:
        kpi(st, "Sensibilité du portefeuille au facteur France", f"{beta_ptf:+.2f}".replace(".", ","),
            "0 = insensible ; 1 = comme le CAC 40 face à l'Europe", delta_color="off")
        sig = sens[sens["t_facteur"].abs() >= 2].sort_values("beta_facteur", ascending=False)
        st.caption(f"{len(sig)} lignes sur {len(sens)} ont une sensibilité statistiquement significative (t d'au moins 2 en valeur absolue). "
                   "Les autres barres sont du bruit : ne les interprète pas.")
        tableau(sens.sort_values("beta_facteur", ascending=False)[["nom", "poids", "beta_facteur", "t_facteur"]].head(8).style.format({
            "poids": lambda v: pct(v, 1), "beta_facteur": lambda v: f"{v:+.2f}".replace(".", ","), "t_facteur": lambda v: f"{v:.1f}".replace(".", ",")}),
            hide_index=True, column_config={"nom": "Les plus exposées", "poids": "Poids", "beta_facteur": "Sensibilité", "t_facteur": "t"})

    # ---------- 4. épisodes
    st.markdown("### 4. Ce que les épisodes réels ont fait à ces titres")
    lignes_ep, detail = [], {}
    for nom, (a, b) in EPISODES_POL.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        fen = adj_eur.loc[(adj_eur.index >= a) & (adj_eur.index <= b)]
        fi = pd.concat([ind["fr"], ind["eu"]], axis=1, keys=["fr", "eu"]).loc[a:b].ffill()
        if len(fen) < 2 or len(fi) < 2:
            continue
        ret = fen.ffill().iloc[-1] / fen.bfill().iloc[0] - 1
        dispo = [t for t in poids.index if t in ret.index and pd.notna(ret[t])]
        if not dispo:
            continue
        w = poids.reindex(dispo)
        rf, re_ = float(fi["fr"].iloc[-1] / fi["fr"].iloc[0] - 1), float(fi["eu"].iloc[-1] / fi["eu"].iloc[0] - 1)
        lignes_ep.append({"episode": nom, "portefeuille": float((w * ret[dispo]).sum() / w.sum() * investi), "cac": rf, "europe": re_,
                          "facteur": (1 + rf) / (1 + re_) - 1, "couverture": float(w.sum() / investi) if investi else np.nan})
        detail[nom] = pd.DataFrame({"nom": pf["nom"].reindex(dispo), "poids": w, "rendement": ret[dispo]})
    ep = pd.DataFrame(lignes_ep)
    if ep.empty:
        st.info("L'historique disponible ne couvre pas ces épisodes.")
    else:
        tableau(ep.style.format({c_: (lambda v: pct(v, 1, True)) for c_ in ["portefeuille", "cac", "europe", "facteur"]} | {"couverture": lambda v: pct(v, 0)})
                .map(signe_couleur, subset=["portefeuille", "cac", "europe", "facteur"]),
                hide_index=True, column_config={"episode": "Épisode", "portefeuille": "Ton portefeuille actuel", "cac": "CAC 40", "europe": ind["nom_eu"],
                                                "facteur": "Facteur France", "couverture": "Poids couvert"})
        st.caption("Tes poids d'aujourd'hui appliqués aux rendements de l'époque. « Poids couvert » : part du portefeuille dont les titres étaient déjà cotés ; "
                   "les autres reçoivent le rendement moyen. Un seul épisode de chaque type : c'est une illustration, pas une loi.")
        choix_ep = st.selectbox("Détail par ligne", list(detail), key="pol_ep")
        dd = detail[choix_ep].sort_values("rendement")
        fig = go.Figure(go.Bar(x=dd["rendement"], y=dd["nom"], orientation="h", marker_color=[C_BON if v >= 0 else C_MAUVAIS for v in dd["rendement"]],
                               hovertemplate="%{y} : %{x:+.1%}<extra></extra>"))
        fig.update_layout(height=max(300, 20 * len(dd)), margin=dict(t=10, l=0, r=10, b=0), xaxis_tickformat="+.0%")
        graphe(fig, key="pol_detail")

    # ---------- 5. ce que ça donnerait
    st.markdown("### 5. Ce que ça donnerait sur ton portefeuille")
    intro("Le raisonnement : une élection est un événement à deux issues, <b>déjà en partie dans les cours</b>. Si le marché donne 30 % au scénario de tension, "
          "30 % du choc est déjà payé. Les cours ne bougent donc que sur la <b>surprise</b> : la probabilité qui change, ou le résultat qui tombe.")
    ref = float(ep.loc[ep["episode"].str.contains("première semaine"), "facteur"].iloc[0]) if len(ep) and ep["episode"].str.contains("première semaine").any() else -0.05
    defaut = int(np.clip(round(min(ref, -0.01) * 100), -20, -1))
    c = st.columns(2)
    choc = c[0].slider("Écart total sur le facteur France entre les deux issues (%)", -20, -1, defaut, 1, key="pol_choc",
                       help="Ce que le CAC 40 perdrait face à l'Europe si l'on passait d'un scénario de tension exclu à un scénario certain. "
                            "Par défaut : le mouvement de la première semaine de juin 2024. C'est un minimum, car la probabilité n'était pas passée de 0 à 100 % cette semaine-là.") / 100
    c[1].markdown(f"Probabilité retenue : **{pct(p, 0)}**. Sensibilité du portefeuille : **{f"{beta_ptf:+.2f}".replace(".", ",")}**."
                  + (f" Écart implicite mesuré par les cotes : **{pct(reac['pente'], 1, True)}**." if reac and abs(reac["t"]) >= 2.6 else ""))
    pb = A.prix_binaire(p, choc, beta_ptf)
    eur_ = lambda v: "—" if pd.isna(total) else f"{v * total:+,.0f} €".replace(",", " ")
    c = st.columns(3)
    kpi(c[0], "Si la probabilité monte de 10 points", pct(pb["dix_points"], 2, True), eur_(pb["dix_points"]))
    kpi(c[1], "Si le scénario de tension se réalise", pct(pb["realise"], 2, True), eur_(pb["realise"]))
    kpi(c[2], "S'il est écarté", pct(pb["ecarte"], 2, True), eur_(pb["ecarte"]))
    st.caption("Ces chiffres ne comptent que l'effet « France contre Europe ». Un choc politique fait souvent baisser aussi l'ensemble des actions européennes : "
               "l'effet total peut être plus fort. Pondéré par les probabilités, le résultat attendu est nul par construction : ce n'est pas une prévision de gain, "
               "c'est une mesure de ton exposition.")

    st.markdown("### Comment lire cette page")
    st.markdown(
        "- **À quoi elle sert** : comprendre si une baisse de tes titres vient de la politique (section 2), savoir quelles lignes y sont exposées (section 3), "
        "et chiffrer ce que tu risques si le marché se met à croire davantage au scénario de tension (section 5).\n"
        "- **Sur quoi reposent les chiffres** : la sensibilité vient d'une régression sur deux ans de cours ; le choc vient d'un épisode réel (juin 2024) ou, s'il est mesurable, "
        "du lien observé entre les cotes et les cours ; la probabilité ne fait que répartir ce choc entre ce qui est déjà payé et ce qui reste à venir.\n"
        "- **Les limites** : un ou deux épisodes seulement ; des sensibilités instables dans le temps ; des cotes peu liquides loin du vote ; deux issues seulement, "
        "alors que la réalité en a davantage ; et le choix des « issues de tension », qui est le tien.\n"
        "- **Ce qu'il ne faut pas en faire** : un pari sur le résultat. Si tu juges ton exposition trop forte, la réponse est de réduire les lignes les plus sensibles, "
        "pas de deviner qui va gagner.\n"
        "- **Sources** : les marchés de prédiction agrègent bien l'information (Arrow et al., 2008 ; Berg, Nelson et Rietz, 2008), avec des biais connus "
        "sur les issues peu probables et les marchés peu liquides. L'effet du risque politique sur les cours est documenté par Pástor et Veronesi (2013) et Kelly, Pástor et Veronesi (2016).")

# ================================================================== PORTEFEUILLE MODÈLE
OR_TICKER = "GOLD.PA"   # Amundi Physical Gold (ETC coté à Paris, en euros)
MODES_MODELE = {"Diversification maximale": "mdp", "Variance minimale": "minvar", "Parité de risque": "erc", "Poids égaux": "egal"}


@st.cache_data(ttl=6 * 3600, show_spinner="Test hors échantillon du portefeuille modèle…")
def modele_backtest_cache(prix, sect, plac, n, ps, mode, borne, forces=()):
    return A.backtest_modele(prix, dict(sect), dict(plac), n, dict(ps), mode, borne, forces=list(forces))


@st.cache_data(ttl=6 * 3600, show_spinner="Sélection refaite sur 200 scénarios de l'année écoulée…")
def modele_robustesse_cache(prix, sect, plac, n, ps, mode, borne, forces=()):
    return A.robustesse_modele(prix, prix.index[-1], dict(sect), dict(plac), n, dict(ps), mode, borne, list(forces), n_scen=200)


@st.cache_data(ttl=6 * 3600, show_spinner="Simulation de 5 000 trajectoires jusqu'à la fin du concours…")
def modele_horizon_cache(rendements, jours, centrer):
    return A.simulation_horizon(rendements, jours, n_sim=5000, centrer_sur="Euronext 100" if centrer else None)


@st.cache_data(ttl=6 * 3600, show_spinner="Comparaison des quatre pondérations…")
def modele_ponderations_cache(prix, sect, plac, n, ps, borne, forces=()):
    """Les quatre pondérations côte à côte : risque prévu aujourd'hui et résultat hors échantillon."""
    sig = A.signaux_modele(prix, prix.index[-1])
    titres, _ = A.selection_modele(sig, dict(sect), dict(plac), n, dict(ps), forces=list(forces))
    cov = prix[titres].pct_change(fill_method=None).iloc[-252:].cov().values * 252
    lignes = []
    for lib, m in MODES_MODELE.items():
        w = A.poids_modele(prix, prix.index[-1], titres, dict(sect), dict(plac), m, borne).values
        v = float(np.sqrt(w @ cov @ w))
        bt = modele_backtest_cache(prix, sect, plac, n, ps, m, borne, forces)
        b = A.bilan_serie(bt["r"]) if bt else {"rend": np.nan, "vol": np.nan, "sharpe": np.nan, "baisse": np.nan}
        lignes.append({"ponderation": lib, "mode": m, "vol_prev": v, "dr": float(w @ np.sqrt(np.diag(cov)) / v) if v > 0 else np.nan,
                       "hhi": float((w ** 2).sum()), "max": float(w.max()), "rend": b["rend"], "vol": b["vol"], "sharpe": b["sharpe"], "baisse": b["baisse"]})
    return pd.DataFrame(lignes)


@st.cache_data(ttl=6 * 3600, show_spinner="Tirage des portefeuilles au hasard, aux mêmes dates…")
def modele_singes_cache(prix, sect, plac, dates, n, n_singes=1000):
    return A.singes_modele(prix, dict(sect), dict(plac), list(dates), n, n_singes=n_singes)


@st.cache_data(ttl=6 * 3600, show_spinner="Bootstrap, Sharpe dégonflé et stabilité…")
def modele_confirm_cache(r, r_egal, r_bench, singes, n_essais):
    return A.confirmations_modele(r, r_egal, r_bench, singes, n_essais=n_essais)


@st.cache_data(ttl=6 * 3600, show_spinner="Variantes du modèle (autres réglages)…")
def modele_variantes_cache(prix, sect, plac, n, ps, mode, borne, forces=()):
    base = dict(ps)
    var = [(f"{max(8, n - 6)} lignes", dict(n=max(8, n - 6))), (f"{n + 6} lignes", dict(n=n + 6))]
    var += [(f"Pondération : {lib.lower()}", dict(mode=m)) for lib, m in MODES_MODELE.items() if m != mode and m != "erc"]
    var += [("Momentum seul", dict(ps={"momentum": 1, "faible_vol": 0, "haut52": 0})),
            ("Faible volatilité seule", dict(ps={"momentum": 0, "faible_vol": 1, "haut52": 0})),
            ("Momentum, volatilité et plus haut à parts égales", dict(ps={"momentum": 1, "faible_vol": 1, "haut52": 1})),
            ("Asymétrie de bêta seule", dict(ps={"momentum": 0, "faible_vol": 0, "haut52": 0, "asym_beta": 1})),
            ("Alpha seul", dict(ps={"momentum": 0, "faible_vol": 0, "haut52": 0, "alpha": 1}))]
    lignes = []
    for nom, ch in var:
        if "ps" in ch and {k: v for k, v in ch["ps"].items() if v} == {k: v for k, v in base.items() if v}:
            continue
        bt = A.backtest_modele(prix, dict(sect), dict(plac), ch.get("n", n), ch.get("ps", base), ch.get("mode", mode), borne, forces=list(forces))
        if bt is None:
            continue
        m, e = A.bilan_serie(bt["r"]), A.bilan_serie(bt["r_egal"])
        lignes.append({"variante": nom, "rend": m["rend"], "sharpe": m["sharpe"], "ecart": m["rend"] - e["rend"], "bat": m["sharpe"] > e["sharpe"]})
    return pd.DataFrame(lignes)


def page_modele():
    st.subheader("Portefeuille modèle : un fil conducteur en quatre étapes")
    intro("Le site construit un portefeuille <b>sans aucun jugement</b>, puis te fait suivre un ordre strict : "
          "<b>1.</b> construire et vérifier que la sélection est solide ; <b>2.</b> simuler ce que les prochains mois peuvent donner ; "
          "<b>3.</b> tester si le modèle se distingue du hasard ; <b>4.</b> seulement si c'est convaincant, approfondir titre par titre avec les autres pages. "
          "<b>Ce n'est pas un pilote automatique</b> : un modèle plus poussé ne devine pas l'avenir, il évite seulement de se fier à un accident des données.")
    # Un formulaire : on règle tout, puis un seul calcul. Sans lui, chaque curseur déplacé relançait le test complet.
    with st.form("mod_reglages", border=False):
        c = st.columns([1.2, 1, 1.2, 1])
        large = c[0].radio("Univers", ["Ma liste de départ", "Univers large"], horizontal=True, key="mod_univ") == "Univers large"
        n = c[1].slider("Nombre de lignes", 12, 40, 26, key="mod_n")
        lib_mode = c[2].selectbox("Pondération", list(MODES_MODELE), key="mod_mode")
        borne = c[3].slider("Poids maximal par ligne (%)", 5, 10, 10, key="mod_borne") / 100
        c = st.columns(5)
        ps = {"alpha": c[4].slider("Poids de l'alpha", 0.0, 2.0, 0.0, 0.5, key="mod_w_alpha",
                                   help="Ce que le titre a gagné sur 12 mois (hors dernier mois) une fois retiré ce que son bêta explique, rapporté à son incertitude. "
                                        "Un titre qui a monté seulement parce qu'il est très sensible au marché a un alpha nul. C'est le « momentum résiduel » "
                                        "(Blitz, Huij et Martens, 2011), ici avec le marché comme seul facteur. Mets les autres curseurs à 0 pour sélectionner sur l'alpha seul."),
              "momentum": c[0].slider("Poids du momentum 12-1", 0.0, 2.0, 1.0, 0.5, key="mod_w_mom"),
              "faible_vol": c[1].slider("Poids de la faible volatilité", 0.0, 2.0, 1.0, 0.5, key="mod_w_vol"),
              "haut52": c[2].slider("Poids du plus haut sur 52 semaines", 0.0, 2.0, 0.0, 0.5, key="mod_w_h52"),
              "asym_beta": c[3].slider("Poids de l'asymétrie de bêta", 0.0, 2.0, 0.0, 0.5, key="mod_w_asym",
                                       help="Bêta les jours de hausse moins bêta les jours de baisse, face à la moyenne de l'univers. Un titre qui monte plus qu'il ne baisse "
                                            "avec le marché a une asymétrie positive. Signal très bruité (chaque bêta repose sur la moitié des séances) : il est à 0 par défaut. "
                                            "Active-le et laisse les confirmations dire s'il apporte quelque chose.")}
        avec_or = st.toggle("Inclure l'or (Amundi Physical Gold, GOLD.PA) dans le portefeuille", key="mod_or",
                            help="L'or n'est pas choisi pour son score : il est imposé comme amortisseur, parce qu'il ne suit pas les actions. "
                                 "C'est la pondération (fondée sur le risque) qui décide de son poids, dans la limite du poids maximal par ligne. "
                                 "Dans le test sur le passé, il n'entre qu'à partir du moment où il a un an de cours.")
        st.form_submit_button("Appliquer les réglages", type="primary",
                              help="Règle tout ce que tu veux, puis clique une seule fois : le test hors échantillon ne se relance qu'à ce moment-là.")
    if sum(ps.values()) == 0:
        st.warning("Donne un poids à au moins un signal.")
        return
    mode = MODES_MODELE[lib_mode]
    pu, bu, u = prix_univers(not large)
    if pu is not None and not pu.empty:
        # jours sans cotation (lignes créées par les devises la nuit, jours fériés) : sans ce nettoyage, aucun titre n'a de cours « aujourd'hui »
        pu = A.nettoyer_prix(pu)
        pu = pu.loc[:, pu.notna().sum() > 0]
    forces = ()
    if avec_or and pu is not None and not pu.empty:
        _, ag = historique((OR_TICKER,))
        if ag is not None and OR_TICKER in ag.columns and ag[OR_TICKER].notna().sum() > 300:
            pu = pu.join(ag[OR_TICKER].reindex(pu.index.union(ag.index)).ffill(limit=5).reindex(pu.index).rename(OR_TICKER), how="left")
            u = pd.concat([u, pd.DataFrame({"nom": "Or physique (Amundi)", "secteur": "Or", "place": "Paris", "univers": "liste"}, index=[OR_TICKER])])
            forces = (OR_TICKER,)
        else:
            st.warning("Les cours de l'or (GOLD.PA) ne sont pas disponibles pour l'instant : le portefeuille est construit sans lui.")
    if pu is None or pu.empty or len(pu) < 300:
        st.info("Historique de l'univers indisponible pour l'instant.")
        return
    sect = tuple(sorted((t, str(u.loc[t, "secteur"])) for t in pu.columns))
    plac = tuple(sorted((t, str(u.loc[t, "place"])) for t in pu.columns))
    ps_t = tuple(sorted(ps.items()))
    essais = st.session_state.setdefault("mod_essais", set())
    cle_reglages = (large, n, mode, borne, ps_t, forces)
    essais.add(cle_reglages)
    etat = st.session_state
    barre = st.empty()          # remplie en fin de page, une fois connus les boutons cliqués pendant ce passage
    pastille = lambda num, nom, ok: (f'<div style="flex:1;min-width:150px;padding:9px 12px;border-radius:10px;border:1px solid rgba(128,128,128,.25);'
                                     f'background:{"rgba(34,197,94,.10)" if ok else "rgba(128,128,128,.05)"}"><span style="font-weight:700">{num}</span> · {nom}'
                                     f'<div style="font-size:.74rem;opacity:.7">{"fait" if ok else "à faire"}</div></div>')
    def dessine_barre(ouverte=False):
        barre.html('<div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 6px">' + pastille(1, "Construire", etat.get("mod_sol") == cle_reglages)
                   + pastille(2, "Simuler la suite", etat.get("mod_sim") == cle_reglages) + pastille(3, "Tester la chance", etat.get("mod_lance") == cle_reglages)
                   + pastille(4, "Approfondir", ouverte) + "</div>")
    dessine_barre()

    # ---------- 1. la proposition du jour
    st.markdown("### Étape 1 · Construire le portefeuille")
    sig = A.signaux_modele(pu, pu.index[-1])
    titres, score = A.selection_modele(sig, dict(sect), dict(plac), n, ps, forces=list(forces))
    if len(titres) < 3:
        st.info("Pas assez de titres avec un an d'historique pour construire le modèle.")
        return
    w = A.poids_modele(pu, pu.index[-1], titres, dict(sect), dict(plac), mode, borne)
    prop = pd.DataFrame({"nom": u["nom"].reindex(titres), "secteur": u["secteur"].reindex(titres), "place": u["place"].reindex(titres),
                         "poids": w, "score": score.reindex(titres), "momentum": sig["momentum"].reindex(titres), "vol": sig["vol"].reindex(titres),
                         "asym": sig["asym_beta"].reindex(titres), "alpha": sig["alpha"].reindex(titres)})
    prop["detenu"] = ["Oui" if t in pf.index else "—" for t in prop.index]
    prop = prop.sort_values("poids", ascending=False)
    Rr = pu[titres].pct_change(fill_method=None).iloc[-252:]
    cov = Rr.cov().values * 252
    vol_p = float(np.sqrt(w.values @ cov @ w.values))
    dr = float((w.values @ np.sqrt(np.diag(cov))) / vol_p) if vol_p > 0 else np.nan
    h_, n_eff = A.hhi(w)
    communs = [t for t in titres if t in pf.index]
    c = st.columns(5)
    kpi(c[0], "Volatilité prévue", pct(vol_p, 1), "sur un an de cours", delta_color="off")
    kpi(c[1], "Ratio de diversification", f"{dr:.2f}".replace(".", ","), f"{dr ** 2:.1f} paris indépendants".replace(".", ","), delta_color="off")
    kpi(c[2], "HHI", pct(h_, 1), f"{n_eff:.1f} lignes effectives".replace(".", ","), delta_color="off")
    kpi(c[3], "Secteurs couverts", f"{prop.loc[prop['secteur'] != 'Or', 'secteur'].nunique()} sur 11", f"plus gros : {pct(prop.groupby('secteur')['poids'].sum().max(), 0)}", delta_color="off")
    kpi(c[4], "En commun avec toi", f"{len(communs)} titres", f"sur {len(titres)} dans le modèle", delta_color="off")
    tableau(prop.style.format({"poids": lambda v: pct(v, 1), "score": lambda v: f"{v:+.2f}".replace(".", ","), "momentum": lambda v: pct(v, 0, True),
                               "vol": lambda v: pct(v, 0), "asym": lambda v: "—" if pd.isna(v) else f"{v:+.2f}".replace(".", ","),
                               "alpha": lambda v: "—" if pd.isna(v) else f"{v:+.1f}".replace(".", ",")}).map(signe_couleur, subset=["momentum", "asym", "alpha"])
            .map(lambda v: f"color:{C_BON};font-weight:600" if v == "Oui" else "", subset=["detenu"]),
            hide_index=True, column_config={"nom": "Titre", "secteur": "Secteur", "place": "Place", "poids": "Poids", "score": "Score",
                                            "momentum": "Momentum 12-1", "vol": "Volatilité", "asym": "Asym. bêta", "alpha": "Alpha (t)", "detenu": "Détenu"})
    if mode in ("mdp", "minvar"):
        st.caption("Ces deux pondérations concentrent les poids sur les titres les moins corrélés : le ratio de diversification monte, mais le HHI se dégrade "
                   f"(poids égaux : {pct(1 / len(titres), 1)}). Les deux sont des critères du concours : « Parité de risque » et « Poids égaux » sont des compromis.")
    idees = [t for t in titres if t not in pf.index]
    absents = [t for t in pf.index if t not in titres]
    g, d = st.columns(2)
    with g:
        st.markdown(f"**Dans le modèle, pas chez toi ({len(idees)})** : les idées à étudier si les tests sont concluants.")
        st.code(", ".join(idees) if idees else "aucun", language=None)
        st.caption("Colle cette liste dans les tickers libres de « Résultats et qualité » pour voir leurs fondamentaux.")
    with d:
        st.markdown(f"**Chez toi, pas dans le modèle ({len(absents)})** : pas une raison de vendre, mais des lignes que seuls tes arguments justifient.")
        st.caption(", ".join(str(pf.loc[t, "nom"]) for t in absents) if absents else "aucune")

    # ---------- 1 bis. solidité de la sélection
    st.markdown("**La sélection est-elle solide ?**")
    st.caption("Le classement repose sur une seule année de cours. Si cette année s'était déroulée un peu autrement, aurait-on choisi les mêmes titres ? "
               "Le site remélange 200 fois les mois de l'année écoulée (les mêmes jours pour tous les titres) et refait la sélection à chaque fois (Michaud, 1998).")
    if st.button("Tester la solidité sur 200 scénarios", key="mod_sol_go"):
        etat["mod_sol"] = cle_reglages
    rob = None
    if etat.get("mod_sol") == cle_reglages:
        rob = modele_robustesse_cache(pu, sect, plac, n, ps_t, mode, borne, forces)
    if rob is not None:
        fr = rob["frequence"]
        libres = [t for t in titres if t not in forces]
        solides, fragiles = [t for t in libres if fr.get(t, 0) >= 0.8], [t for t in libres if fr.get(t, 0) < 0.5]
        stab = float(fr.reindex(libres).mean()) if libres else np.nan
        c = st.columns(4)
        kpi(c[0], "Stabilité de la sélection", pct(stab, 0), "fréquence moyenne des titres retenus", delta_color="off")
        kpi(c[1], "Titres solides", f"{len(solides)} sur {len(libres)}", "retenus dans 80 % des scénarios ou plus", delta_color="off")
        kpi(c[2], "Titres fragiles", f"{len(fragiles)}", "retenus dans moins d'un scénario sur deux", delta_color="off")
        ecart_w = float((rob["poids_moyens"].reindex(titres) - w.reindex(titres)).abs().sum() / 2)
        kpi(c[3], "Poids déplacés par les scénarios", pct(ecart_w, 0), "part du portefeuille qui change", delta_color="off")
        ts = pd.DataFrame({"nom": u["nom"].reindex(titres), "freq": fr.reindex(titres), "poids": w.reindex(titres),
                           "poids_moy": rob["poids_moyens"].reindex(titres), "disp": rob["poids_ecart"].reindex(titres)}).sort_values("freq")
        ts["avis"] = ["Imposé" if t in forces else ("Solide" if f_ >= 0.8 else ("Fragile" if f_ < 0.5 else "Moyen")) for t, f_ in ts["freq"].items()]
        g, d = st.columns([3, 2])
        with g:
            fig = go.Figure(go.Bar(x=ts["freq"], y=ts["nom"], orientation="h",
                                   marker_color=[C_NEUTRE if a_ == "Imposé" else (C_BON if a_ == "Solide" else (C_MAUVAIS if a_ == "Fragile" else C_ATTENTION)) for a_ in ts["avis"]],
                                   hovertemplate="%{y} : retenu dans %{x:.0%} des scénarios<extra></extra>"))
            fig.add_vline(x=0.8, line_dash="dot", line_color=C_NEUTRE); fig.add_vline(x=0.5, line_dash="dot", line_color=C_NEUTRE)
            fig.update_layout(height=max(320, 19 * len(ts)), margin=dict(t=36, l=0, r=10, b=0), xaxis_tickformat=".0%", xaxis_range=[0, 1.02],
                              title=dict(text="Fréquence de sélection de chaque titre retenu", font_size=13))
            graphe(fig, key="mod_sol")
        with d:
            rempl = fr.drop(index=[t for t in titres if t in fr.index]).sort_values(ascending=False).head(6)
            rempl = rempl[rempl > 0.2]
            st.markdown("**Les remplaçants** : absents de la sélection, mais souvent retenus dans les scénarios.")
            if len(rempl):
                tableau(pd.DataFrame({"nom": u["nom"].reindex(rempl.index), "freq": rempl.values}, index=rempl.index).style.format({"freq": lambda v: pct(v, 0)}),
                        hide_index=True, column_config={"nom": "Titre", "freq": "Fréquence de sélection"})
            else:
                st.caption("Aucun titre écarté ne revient souvent : la frontière de la sélection est nette.")
            st.markdown("**Les fragiles** : " + (", ".join(str(u["nom"].get(t, t)) for t in fragiles) if fragiles else "aucun") + ".")
        if stab >= 0.75 and ecart_w <= 0.15:
            st.success(f"Sélection solide : en moyenne, un titre retenu l'est dans {pct(stab, 0)} des scénarios, et les poids bougent peu.")
        elif stab >= 0.6:
            st.warning(f"Sélection moyennement solide ({pct(stab, 0)}). Les titres « fragiles » doivent leur place à quelques semaines de cours : ne t'y attache pas.")
        else:
            st.error(f"Sélection fragile ({pct(stab, 0)}) : avec une année à peine différente, on aurait choisi un autre portefeuille. Les signaux retenus séparent mal les titres.")
        st.caption("Limite : remélanger les mois garde les mêmes rendements d'ensemble ; le test mesure la sensibilité au calendrier et aux semaines extrêmes, "
                   "pas ce qui se passerait dans une année réellement différente.")

    # ---------- 2. simuler la suite
    st.markdown("### Étape 2 · Simuler la suite jusqu'à la fin du concours")
    intro("Des milliers de trajectoires possibles, fabriquées en remettant bout à bout des morceaux de deux semaines tirés des trois dernières années. "
          "Le modèle, ton portefeuille et l'indice reçoivent <b>les mêmes jours</b> : on compare donc leur comportement dans les mêmes mondes. "
          "C'est une mesure de <b>dispersion</b>, pas une prévision.")
    c = st.columns([1, 2])
    fin = c[0].date_input("Fin du concours", dt.date(2027, 4, 30), key="mod_fin", format="DD/MM/YYYY",
                          help="Vérifie la date exacte dans le règlement : elle fixe le nombre de séances simulées.")
    hyp = c[1].radio("Hypothèse de rendement", ["Aucun avantage : tout le monde a le rendement moyen de l'indice", "Rendements passés de chacun"], key="mod_hyp",
                     help="Le modèle choisit des titres qui ont bien marché : rejouer leurs rendements passés le flatte forcément. "
                          "Avec « aucun avantage », seuls le risque et la diversification distinguent les portefeuilles : c'est l'hypothèse honnête.")
    if st.button("Lancer la simulation", key="mod_sim_go"):
        etat["mod_sim"] = cle_reglages
    if etat.get("mod_sim") == cle_reglages:
        jours_h = int(np.busday_count(dt.datetime.now(TZ).date(), fin))
        Rm = pu[titres].pct_change(fill_method=None).iloc[-756:]
        ok_c = [t for t in titres if Rm[t].notna().sum() > 250]
        wm = w.reindex(ok_c) / w.reindex(ok_c).sum()
        series = {"Portefeuille modèle": Rm[ok_c].fillna(0.0) @ wm}
        miens = [t for t in pf.index if t in adj_eur.columns]
        if miens and float(poids.reindex(miens).sum()) > 0:
            Rt = A.nettoyer_prix(adj_eur[miens]).pct_change(fill_method=None).reindex(Rm.index)
            series["Ton portefeuille"] = Rt.fillna(0.0) @ poids.reindex(miens).fillna(0.0)
        if bu is not None:
            series["Euronext 100"] = bu.ffill().pct_change(fill_method=None).reindex(Rm.index).fillna(0.0)
        Rs = pd.DataFrame(series).iloc[1:]
        centrer = hyp.startswith("Aucun") and "Euronext 100" in Rs.columns
        sim = modele_horizon_cache(Rs, jours_h, centrer) if jours_h >= 5 else None
        if sim is None:
            st.info("Simulation impossible : il faut au moins cinq séances avant la date de fin et six mois d'historique.")
        else:
            tot, creux = sim
            ref = "Euronext 100" if "Euronext 100" in tot.columns else None
            lignes_s = []
            for nom_ in tot.columns:
                lignes_s.append({"ptf": nom_, "p5": float(tot[nom_].quantile(0.05)), "med": float(tot[nom_].median()), "p95": float(tot[nom_].quantile(0.95)),
                                 "perte": float((tot[nom_] < 0).mean()), "creux": float(creux[nom_].median()),
                                 "bat": float((tot[nom_] > tot[ref]).mean()) if ref and nom_ != ref else np.nan})
            g, d = st.columns([3, 2])
            with g:
                fig = go.Figure()
                for k_, nom_ in enumerate(tot.columns):
                    fig.add_trace(go.Box(x=tot[nom_].clip(tot[nom_].quantile(0.005), tot[nom_].quantile(0.995)), name=nom_, boxpoints=False,
                                         marker_color=[PALETTE[0], PALETTE[1], PALETTE[6]][k_ % 3], hoverinfo="x"))
                fig.add_vline(x=0, line_color=C_NEUTRE, line_width=1)
                fig.update_layout(height=280, margin=dict(t=36, l=0, r=10, b=0), xaxis_tickformat="+.0%", showlegend=False,
                                  title=dict(text=f"Rendement possible sur {jours_h} séances (boîte : la moitié centrale des trajectoires)", font_size=13))
                graphe(fig, key="mod_sim")
            with d:
                if "Ton portefeuille" in tot.columns:
                    kpi(st, "Le modèle finit devant toi dans", pct(float((tot["Portefeuille modèle"] > tot["Ton portefeuille"]).mean()), 0),
                        "des trajectoires (50 % = pile ou face)", delta_color="off")
                    e5 = float(tot["Portefeuille modèle"].quantile(0.05) - tot["Ton portefeuille"].quantile(0.05))
                    kpi(st, "Écart dans le mauvais cas", f"{e5 * 100:+.1f} pt".replace(".", ","), "modèle moins toi, au 5e centile", delta_color="off")
            tableau(pd.DataFrame(lignes_s).style.format({"p5": lambda v: pct(v, 1, True), "med": lambda v: pct(v, 1, True), "p95": lambda v: pct(v, 1, True),
                                                         "perte": lambda v: pct(v, 0), "creux": lambda v: pct(v, 1), "bat": lambda v: "—" if pd.isna(v) else pct(v, 0)})
                    .map(signe_couleur, subset=["p5", "med", "p95"]),
                    hide_index=True, column_config={"ptf": "Portefeuille", "p5": "Mauvais cas (1 sur 20)", "med": "Cas médian", "p95": "Bon cas (1 sur 20)",
                                                    "perte": "Risque de finir en perte", "creux": "Baisse en cours de route", "bat": "Bat l'indice"})
            st.caption(("Hypothèse « aucun avantage » : tous ont le même rendement moyen par jour. Le cas médian d'un portefeuille plus calme ressort quand même un peu plus haut : "
                        "à rendement moyen égal, la volatilité ronge le résultat composé. Regarde surtout le mauvais cas et la baisse en cours de route. " if centrer else
                        "Hypothèse « rendements passés » : le modèle est flatté, puisqu'il a choisi ses titres sur ces mêmes rendements. Ne retiens que l'écart de dispersion. ")
                       + "Limites : les trois dernières années ne contiennent pas toutes les crises possibles, et les poids sont supposés fixes.")
            cm = stress(w, pu, None)
            ct = stress(poids, adj_eur, BENCH_PRINCIPAL)
            if len(cm):
                cr = cm.rename(columns={"portefeuille": "modele"})[["scenario", "modele", "couverture"]]
                if len(ct):
                    cr = cr.merge(ct.rename(columns={"portefeuille": "toi", "reference": "indice"})[["scenario", "toi", "indice"]], on="scenario", how="left")
                cols_c = [c_ for c_ in ["modele", "toi", "indice"] if c_ in cr.columns]
                st.markdown("**Et si une crise passée se reproduisait ?**")
                tableau(cr[["scenario"] + cols_c + ["couverture"]].style.format({c_: (lambda v: "—" if pd.isna(v) else pct(v, 1, True)) for c_ in cols_c} | {"couverture": lambda v: pct(v, 0)})
                        .map(signe_couleur, subset=cols_c),
                        hide_index=True, column_config={"scenario": "Crise rejouée", "modele": "Portefeuille modèle", "toi": "Ton portefeuille actuel",
                                                        "indice": "Euronext 100", "couverture": "Poids couvert"})
                st.caption("Les poids d'aujourd'hui appliqués aux cours de l'époque. « Poids couvert » concerne le modèle : la part de ses titres déjà cotés à ce moment-là.")

    def _etape_chance():
        # ---------- 2. test hors échantillon
        st.markdown("### Étape 3 · Tester la chance : ce modèle aurait-il marché ?")
        intro("À chaque fin de mois du passé, le site reconstruit le portefeuille avec <b>les seules données connues ce jour-là</b>, le détient un mois, "
              "puis recommence. Aucun réglage n'a vu le futur. La référence honnête est l'<b>univers à poids égaux</b> : mêmes titres, aucun choix.")
        if st.button("Lancer les tests de chance (environ une minute)", type="primary", key="mod_go"):
            st.session_state["mod_lance"] = cle_reglages
        if st.session_state.get("mod_lance") != cle_reglages:
            if st.session_state.get("mod_lance"):
                st.caption("Les réglages ont changé : relance les tests pour ce nouveau modèle.")
            return None
        bt = modele_backtest_cache(pu, sect, plac, n, ps_t, mode, borne, forces)
        if bt is None or len(bt["r"]) < 120:
            st.info("Historique trop court pour un test hors échantillon (il faut au moins un an et demi de cours).")
            return None
        r_m, r_e = bt["r"], bt["r_egal"]
        r_b = bu.pct_change(fill_method=None).reindex(r_m.index).fillna(0.0) if bu is not None else None
        bm, be = A.bilan_serie(r_m), A.bilan_serie(r_e)
        bb = A.bilan_serie(r_b) if r_b is not None else None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r_m.index, y=(1 + r_m).cumprod() - 1, name="Portefeuille modèle", line=dict(width=2.4, color=PALETTE[0])))
        fig.add_trace(go.Scatter(x=r_e.index, y=(1 + r_e).cumprod() - 1, name="Univers à poids égaux", line=dict(width=1.8, color=PALETTE[2])))
        if r_b is not None:
            fig.add_trace(go.Scatter(x=r_b.index, y=(1 + r_b).cumprod() - 1, name="Euronext 100", line=dict(width=1.6, color=PALETTE[6], dash="dot")))
        fig.update_layout(height=340, yaxis_tickformat="+.0%", margin=dict(t=10, l=0, r=0, b=0), legend=dict(orientation="h", y=-0.15), hovermode="x unified")
        graphe(fig, key="mod_bt")
        c = st.columns(5)
        kpi(c[0], "Rendement annualisé", pct(bm["rend"], 1, True), f"poids égaux {pct(be['rend'], 1, True)}", delta_color="off")
        kpi(c[1], "Volatilité", pct(bm["vol"], 1), f"poids égaux {pct(be['vol'], 1)}", delta_color="off")
        kpi(c[2], "Ratio de Sharpe", f"{bm['sharpe']:.2f}".replace(".", ","), f"poids égaux {be['sharpe']:.2f}".replace(".", ","), delta_color="off")
        kpi(c[3], "Plus forte baisse", pct(bm["baisse"], 0), f"poids égaux {pct(be['baisse'], 0)}", delta_color="off")
        kpi(c[4], "Rotation mensuelle", pct(bt["rotation"], 0) if pd.notna(bt["rotation"]) else "—", "part du portefeuille renouvelée", delta_color="off")
        st.caption(f"Du {r_m.index.min():%d/%m/%Y} au {r_m.index.max():%d/%m/%Y}, {len(bt['compositions'])} reconstructions mensuelles. "
                   "Attention : l'univers est celui d'aujourd'hui (les titres disparus ou déclassés n'y sont pas), donc tous les rendements absolus sont trop flatteurs. "
                   "C'est pour cela que le modèle est jugé face à l'univers à poids égaux et aux tirages au hasard, qui profitent du même biais.")

        st.markdown("**Les quatre pondérations côte à côte**")
        if st.toggle("Comparer les quatre pondérations (trois tests de plus : compte une à deux minutes)", key="mod_cmp"):
            cmp_ = modele_ponderations_cache(pu, sect, plac, n, ps_t, borne, forces).sort_values("vol_prev")
            tableau(cmp_.drop(columns="mode").style.format({"vol_prev": lambda v: pct(v, 1), "dr": lambda v: f"{v:.2f}".replace(".", ","), "hhi": lambda v: pct(v, 1),
                                                            "max": lambda v: pct(v, 1), "rend": lambda v: pct(v, 1, True), "vol": lambda v: pct(v, 1),
                                                            "sharpe": lambda v: f"{v:.2f}".replace(".", ","), "baisse": lambda v: pct(v, 0)})
                    .apply(lambda r_: ["font-weight:700" if r_["ponderation"] == lib_mode else "" for _ in r_], axis=1),
                    hide_index=True, column_config={"ponderation": "Pondération", "vol_prev": "Vol. prévue", "dr": "Diversif.", "hhi": "HHI",
                                                    "max": "Poids max", "rend": "Rendement", "vol": "Volatilité",
                                                    "sharpe": "Sharpe", "baisse": "Baisse max"})
            st.caption("Du moins risqué au plus risqué, pour les mêmes titres. Les quatre premières colonnes (volatilité prévue, ratio de diversification, HHI, poids maximal) "
                       "décrivent le portefeuille d'aujourd'hui ; les quatre suivantes (rendement annualisé, volatilité, Sharpe, plus forte baisse) viennent du test hors échantillon. En gras : la pondération sélectionnée. Choisis d'abord sur le risque et sur les critères du concours (ratio de diversification, HHI) : "
                       "le rendement passé est la colonne la moins fiable, et retenir la pondération au meilleur Sharpe compte comme un réglage de plus.")

        # ---------- 3. confirmations
        st.markdown("**La batterie de confirmations**")
        st.caption("Chaque test pose une question différente. Un seul test réussi ne prouve rien ; c'est leur accord qui compte.")
        n_conf = len(essais)
        singes = modele_singes_cache(pu, sect, plac, tuple(bt["compositions"].keys()), n)
        variantes = modele_variantes_cache(pu, sect, plac, n, ps_t, mode, borne, forces)
        n_essais = n_conf + len(variantes) + 3          # + les trois autres pondérations du tableau comparatif
        cf = modele_confirm_cache(r_m, r_e, r_b, singes, n_essais)
        X = pd.DataFrame({"Marché": r_b}) if r_b is not None else pd.DataFrame(index=r_m.index)
        wml, bv = wml_cache(pu), bv_cache(pu)
        if len(wml) > 60:
            X["Momentum"] = wml
        if len(bv) > 60:
            X["Basse volatilité"] = bv
        fac = A.regression_multi(r_m, X) if X.shape[1] else None

        oui, non = "Confirmé", "Non confirmé"
        lignes, nb_ok = [], 0
        def ajoute(test, question, resultat, ok):
            nonlocal nb_ok
            nb_ok += int(bool(ok))
            lignes.append({"test": test, "question": question, "resultat": resultat, "verdict": oui if ok else non})
        if "p_rendement" in cf:
            ajoute("Hasard : rendement", "Des portefeuilles tirés au sort aux mêmes dates font-ils aussi bien ?",
                   f"{pct(cf['p_rendement'], 1)} des tirages font au moins aussi bien", cf["p_rendement"] < 0.05)
            ajoute("Hasard : Sharpe", "Et à risque comparable ?", f"{pct(cf['p_sharpe'], 1)} des tirages ont un Sharpe au moins égal", cf["p_sharpe"] < 0.05)
        if "p_alpha" in cf:
            ajoute("Alpha face à l'Euronext 100", "Le rendement non expliqué par le bêta dépasse-t-il le bruit ? (bootstrap par blocs)",
                   f"alpha {pct(cf['alpha_an'], 1, True)} par an, p = {cf['p_alpha']:.3f}".replace(".", ","), cf["p_alpha"] < 0.05)
        if "dsr" in cf:
            ajoute("Sharpe dégonflé", f"Le Sharpe résiste-t-il au fait d'avoir essayé {n_essais} réglages ?",
                   f"probabilité {pct(cf['dsr'], 0)} que le vrai Sharpe soit positif", cf["dsr"] > 0.95)
        if "part_annees" in cf and pd.notna(cf["part_annees"]):
            stable = cf["part_annees"] >= 0.6 and min(cf["moities"]) > 0
            ajoute("Stabilité dans le temps", "Le modèle bat-il l'univers à poids égaux la plupart des années, et dans chaque moitié de la période ?",
                   f"{pct(cf['part_annees'], 0)} des années ; moitiés : {pct(cf['moities'][0], 0, True)} puis {pct(cf['moities'][1], 0, True)}", stable)
        if len(variantes):
            part = float(variantes["bat"].mean())
            ajoute("Robustesse aux réglages", "Les réglages voisins marchent-ils aussi, ou seul celui-ci ?",
                   f"{int(variantes['bat'].sum())} variantes sur {len(variantes)} battent les poids égaux", part >= 0.75)
        total = len(lignes)
        c = st.columns(3)
        kpi(c[0], "Confirmations", f"{nb_ok} sur {total}", "tests réussis", delta_color="off")
        kpi(c[1], "Réglages essayés", f"{n_essais}", f"{n_conf} par toi, {len(variantes)} variantes, 3 pondérations", delta_color="off")
        if fac is not None:
            kpi(c[2], "Alpha une fois les facteurs retirés", pct(fac["alpha"] * 252, 1, True), f"t = {fac['t_alpha']:.1f}".replace(".", ","), delta_color="off")
        if total and nb_ok >= max(total - 1, 1):
            st.success(f"{nb_ok} confirmations sur {total} : le résultat est solide pour un modèle aussi simple. Tu peux passer à l'analyse en profondeur "
                       "des titres proposés (thèse, menace, bilan, valorisation), un par un.")
        elif total and nb_ok >= total / 2:
            st.warning(f"{nb_ok} confirmations sur {total} : résultat fragile. Utilise la liste comme une piste parmi d'autres, sans lui donner de poids particulier.")
        else:
            st.error(f"{nb_ok} confirmations sur {total} : le modèle ne se distingue pas du hasard sur cette période. N'en tire pas d'idées d'achat avec ces réglages. "
                     "Ne multiplie pas les réglages jusqu'à ce que ça passe : chaque essai de plus rend le test plus exigeant (Sharpe dégonflé).")
        st.html(CSS_GLOBAL + '<div class="nw-al">' + "".join(
            f'<div class="a {"ok" if l["verdict"] == oui else "bad"}"><span class="tag" style="min-width:104px;text-align:center;'
            f'{"background:rgba(34,197,94,.14);color:#15803d" if l["verdict"] == oui else ""}">{l["verdict"]}</span>'
            f'<span><b>{html.escape(l["test"])}</b> · {html.escape(l["question"])}<br><span style="opacity:.8">{html.escape(l["resultat"])}</span></span></div>'
            for l in lignes) + "</div>")

        g, d = st.columns(2)
        with g:
            if "singes_total" in cf:
                tot = cf["singes_total"]
                fig = go.Figure(go.Histogram(x=tot, nbinsx=50, marker_color=C_NEUTRE, hovertemplate="%{x:+.0%} : %{y} tirages<extra></extra>"))
                fig.add_vline(x=bm["total"], line_color=PALETTE[0], line_width=3)
                fig.add_annotation(x=bm["total"], y=1, yref="paper", text="Modèle", showarrow=False, yanchor="bottom", font=dict(color=PALETTE[0]))
                fig.update_layout(height=300, margin=dict(t=36, l=0, r=0, b=0), xaxis_tickformat="+.0%", bargap=0.02,
                                  title=dict(text=f"Rendement total de {len(tot)} portefeuilles tirés au sort", font_size=13))
                graphe(fig, key="mod_singes")
        with d:
            if "annees" in cf and len(cf["annees"]):
                an = cf["annees"]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=an.index.astype(str), y=an["m"], name="Modèle", marker_color=PALETTE[0]))
                fig.add_trace(go.Bar(x=an.index.astype(str), y=an["e"], name="Poids égaux", marker_color=PALETTE[2]))
                fig.update_layout(height=300, margin=dict(t=36, l=0, r=0, b=0), yaxis_tickformat="+.0%", barmode="group", legend=dict(orientation="h", y=-0.15),
                                  title=dict(text="Année par année", font_size=13))
                graphe(fig, key="mod_annees")
        if len(variantes):
            st.markdown("**Les variantes testées**")
            tableau(variantes.assign(bat=variantes["bat"].map({True: "Oui", False: "Non"})).style.format({
                "rend": lambda v: pct(v, 1, True), "sharpe": lambda v: f"{v:.2f}".replace(".", ","), "ecart": lambda v: pct(v, 1, True)})
                .map(signe_couleur, subset=["ecart"]),
                hide_index=True, column_config={"variante": "Variante", "rend": "Rendement annualisé", "sharpe": "Sharpe",
                                                "ecart": "Écart avec les poids égaux", "bat": "Bat les poids égaux (Sharpe)"})
        if fac is not None:
            st.markdown("**D'où vient la performance ?**")
            st.caption("Régression du modèle sur le marché et sur les facteurs momentum et basse volatilité construits sur le même univers. "
                       + ", ".join(f"{k} : {v:+.2f}".replace(".", ",") for k, v in fac["betas"].items())
                       + f" ; R² ajusté {pct(fac['r2_adj'], 0)}. Un alpha proche de zéro est le résultat attendu : le modèle ne fait que s'exposer à ces facteurs "
                         "connus. Ce n'est pas un défaut, mais ce n'est pas du talent de sélection non plus.")
        return nb_ok, total

    verdict = _etape_chance()
    dessine_barre()

    # ---------- 4. approfondir
    st.markdown("### Étape 4 · Approfondir, titre par titre")
    if verdict is None:
        st.info("Cette étape s'ouvre après les tests de chance. Tant que le modèle n'a pas montré qu'il se distingue du hasard, étudier ses titres un par un "
                "revient à étudier des titres pris au sort.")
    else:
        nb_ok, total = verdict
        convaincant = bool(total) and nb_ok >= max(total - 1, 1)
        dessine_barre(convaincant)
        if convaincant:
            st.success("Le modèle a passé les tests : ses titres méritent une analyse. L'analyse reste la tienne : le modèle ne voit ni les bilans, ni les résultats, ni les événements à venir.")
        else:
            st.error(f"Arrête-toi ici : {nb_ok} confirmation{'s' if nb_ok > 1 else ''} sur {total}. Rien n'indique que ces titres valent mieux que d'autres. "
                     "Le résultat utile de cette page est justement de ne pas suivre ce modèle.")
        if convaincant or st.checkbox("Voir quand même la suite, à titre d'exploration", key="mod_force"):
            ordre = sorted(idees, key=lambda t: -(rob["frequence"].get(t, 0) if rob is not None else 0))
            st.markdown(f"**À étudier en premier** ({len(ordre)} titres absents de ton portefeuille"
                        + (", les plus solides d'abord" if rob is not None else "") + ") : "
                        + (", ".join(str(u["nom"].get(t, t)) for t in ordre[:12]) if ordre else "aucun") + ".")
            c = st.columns(4)
            if ordre and c[0].button("Voir leurs résultats et leur qualité", key="mod_vers_res", type="primary"):
                st.session_state["res_libres"] = ", ".join(t for t in ordre[:12] if t != OR_TICKER)
                st.switch_page(PAGES["resultats"])
            c[1].page_link(PAGES["candidats"], label="Candidats : les règles d'entrée")
            c[2].page_link(PAGES["valorisation"], label="Valorisation : le prix payé")
            c[3].page_link(PAGES["simulateur"], label="Simulateur : tester l'ordre")
            st.markdown("Pour chaque titre retenu, dans l'ordre : **1.** les bénéfices suivent-ils le cours ? (Résultats et qualité) · **2.** le prix est-il raisonnable ? (Valorisation) · "
                        "**3.** respecte-t-il tes règles ? (Candidats) · **4.** que fait-il à ton risque ? (Simulateur) · **5.** écris la thèse, la menace et la condition de sortie avant d'acheter.")

    st.markdown("### Comment lire cette page")
    st.markdown(
        "- **L'ordre compte** : construire, simuler, tester la chance, et seulement ensuite analyser les titres. Un titre proposé par un modèle non confirmé ne vaut pas mieux qu'un titre pris au hasard.\n"
        "- **Étape 1, la solidité** : un titre retenu dans 90 % des scénarios est un choix du modèle ; un titre à 30 % est un accident du calendrier.\n"
        "- **Étape 2, la simulation** : elle ne dit pas ce qui va arriver, elle dit à quel point les issues possibles sont dispersées. Garde l'hypothèse « aucun avantage ».\n"
        "- **L'or** : imposé comme amortisseur, pas choisi par les signaux. Regarde ce qu'il change au mauvais cas de l'étape 2 et aux crises rejouées.\n"
        "- **Ce que le modèle ne voit pas** : les bilans, les résultats, les révisions des analystes, les événements à venir. Yahoo ne donne pas ces données telles qu'elles "
        "étaient connues dans le passé, donc on ne peut pas les tester honnêtement hors échantillon. Elles interviennent à l'étape suivante, dans « Résultats et qualité » et dans ta fiche de ligne.\n"
        "- **Les limites du test** : l'univers d'aujourd'hui (biais du survivant), pas de frais, des poids fixes dans le mois, quelques années d'historique seulement. "
        "Un résultat confirmé ici reste une présomption, pas une preuve.\n"
        "- **Le piège à éviter** : bouger les curseurs jusqu'à obtenir un beau résultat. Le site compte tes essais et durcit le Sharpe dégonflé en conséquence "
        "(Bailey et López de Prado, 2014).\n"
        "- **Sources** : Jegadeesh et Titman (1993), Blitz et van Vliet (2007), George et Hwang (2004), Choueifaty et Coignard (2008), "
        "DeMiguel, Garlappi et Uppal (2009), Ledoit et Wolf (2004), Kosowski et al. (2006).")

# ================================================================== RÉSULTATS ET QUALITÉ
@st.cache_data(ttl=24 * 3600, show_spinner="Capitalisations boursières (Yahoo)…")
def capitalisations(tickers):
    out = {}
    for t in tickers:
        if MOCK or yf is None:
            import zlib
            out[t] = float(np.exp(np.random.default_rng(zlib.crc32(t.encode())).normal(23.5, 1.0)))
            continue
        try:
            out[t] = float(yf.Ticker(t).fast_info["market_cap"])
        except Exception as e:
            log(f"capitalisation {t} : {type(e).__name__}")
            out[t] = np.nan
    return pd.Series(out, dtype=float)


def _derive_txt(sg, jj):
    if pd.isna(jj):
        return "—"
    if sg == 1:
        return f"↑ encore ~{max(0, int(87 - jj))} j"
    if sg == -1:
        return f"↓ encore ~{max(0, int(87 - jj))} j"
    return "terminée" if jj > 87 else "neutre"


def page_resultats():
    st.subheader("Résultats, révisions et qualité")
    intro("Quatre familles de signaux qui résistent aux études de réplication (Hou, Xue et Zhang, 2020 ; Jensen, Kelly et Pedersen, 2023) : "
          "<b>la surprise de résultats</b> et la dérive qui la suit, <b>les révisions des analystes</b>, <b>la rentabilité brute</b> et "
          "<b>la discipline d'investissement</b> (croissance de l'actif, émissions d'actions). Ils complètent le momentum et la valorisation ; "
          "aucun ne suffit seul pour décider.")
    univ_all = charger_univers()
    c1, c2 = st.columns([3, 2])
    ajout = c1.multiselect("Comparer avec des titres de l'univers", [t for t in univ_all.index if t not in pf.index], key="res_univ",
                           format_func=lambda t: f"{univ_all.loc[t, 'nom']} ({t})", placeholder="Aucun : seulement tes lignes")
    libres = c2.text_input("Ou d'autres tickers Yahoo (séparés par des virgules)", key="res_libres", placeholder="ex. SU.PA, NEXI.MI")
    libres = [x.strip().upper() for x in libres.split(",") if x.strip()]
    tick = list(pf.index) + [t for t in list(ajout) + libres if t not in pf.index]
    tick = list(dict.fromkeys(tick))
    sect = tuple(sorted((t, str(pf.loc[t, "secteur"]) if t in pf.index else str(univ_all["secteur"].get(t, ""))) for t in tick))
    res_f, maj = lire_resultats()
    deja = not res_f.empty and all(t in res_f.index for t in tick)
    if st.button("Charger ou actualiser les données Yahoo", type="secondary" if deja else "primary", key="res_btn"):
        st.session_state["res_charge"] = True
    if st.session_state.get("res_charge"):
        res = donnees_resultats(tuple(tick), sect)
        maj = dt.datetime.now().isoformat() if maj is None else maj
    elif deja:
        res = res_f.reindex(tick)
    else:
        st.info("Clique sur le bouton : la première fois, Yahoo met une à deux minutes pour tes lignes. Les données sont ensuite gardées 24 heures "
                "et alimentent aussi les alertes du tableau de bord.")
        return
    auj = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    rows = []
    for t in tick:
        r = res.loc[t] if t in res.index else pd.Series(dtype=object)
        g = lambda k: pd.to_numeric(r.get(k), errors="coerce")
        sg, jj = A.signal_pead(g("surprise"), r.get("date_pub"), auj)
        gp, ca, em, r90, r30, nr = g("gp_a"), g("croiss_actif"), g("emission"), g("rev90"), g("rev30"), g("net_rev30")
        plus = int(sum([sg == 1, pd.notna(r90) and r90 > 0.02, pd.notna(gp) and gp >= 0.30, pd.notna(ca) and ca < 0.05, pd.notna(em) and em < -0.005]))
        moins = int(sum([sg == -1, pd.notna(r90) and r90 < -0.02, pd.notna(gp) and gp < 0.10, pd.notna(ca) and ca > 0.20, pd.notna(em) and em > 0.02]))
        nom = pf.loc[t, "nom"] if t in pf.index else (univ_all.loc[t, "nom"] if t in univ_all.index else t)
        dp = r.get("date_pub")
        rows.append({"ticker": t, "nom": nom if t in pf.index else f"{nom} (candidat)", "detenu": t in pf.index,
                     "surprise": g("surprise"), "date_pub": pd.Timestamp(dp) if isinstance(dp, str) and dp else pd.NaT,
                     "derive": _derive_txt(sg, jj), "pead": sg, "rev30": r30, "rev90": r90, "net_rev30": nr,
                     "gp_a": gp, "croiss_actif": ca, "emission": em, "plus": plus, "moins": moins})
    df = pd.DataFrame(rows)
    mine = df[df["detenu"]]
    c = st.columns(4)
    kpi(c[0], "Dérive post-annonce favorable", f"{int((mine['pead'] == 1).sum())} lignes", f"défavorable : {int((mine['pead'] == -1).sum())}", delta_color="off")
    kpi(c[1], "Prévisions relevées sur 90 j", f"{int((mine['rev90'] > 0.02).sum())} lignes", f"abaissées : {int((mine['rev90'] < -0.02).sum())}", delta_color="off")
    gp_med = mine["gp_a"].median()
    kpi(c[2], "Rentabilité brute médiane", pct(gp_med, 0) if pd.notna(gp_med) else "—", "marge brute ÷ actif, hors financières", delta_color="off")
    couv = float(mine[["surprise", "rev90", "gp_a"]].notna().mean().mean()) if len(mine) else np.nan
    kpi(c[3], "Données disponibles", pct(couv, 0) if pd.notna(couv) else "—",
        f"mises à jour le {pd.Timestamp(maj):%d/%m à %H:%M}" if maj else "", delta_color="off")

    vue = df.sort_values(["detenu", "moins", "plus"], ascending=[False, False, True])
    f_pct = lambda d_=0, sg_=True: (lambda v: "—" if pd.isna(v) else pct(v, d_, sg_))
    vue = vue.copy()
    vue["surprise_txt"] = [("—" if pd.isna(v) else pct(v, 1, True) + (f" ({d_:%d/%m})" if pd.notna(d_) else "")) for v, d_ in zip(vue["surprise"], vue["date_pub"])]
    vue["bilan"] = [f"{p_} + / {m_} −" for p_, m_ in zip(vue["plus"], vue["moins"])]
    tableau(vue[["nom", "surprise_txt", "derive", "rev90", "net_rev30", "gp_a", "croiss_actif", "emission", "bilan"]].style.format({
        "rev90": f_pct(1),
        "net_rev30": lambda v: "—" if pd.isna(v) else f"{int(v):+d}", "gp_a": f_pct(0, False), "croiss_actif": f_pct(0), "emission": f_pct(1),
        })
        .map(lambda v: f"color:{C_BON}" if isinstance(v, str) and v.startswith("+") else (f"color:{C_MAUVAIS}" if isinstance(v, str) and v.startswith("-") else ""), subset=["surprise_txt"])
        .map(signe_couleur, subset=["rev90", "net_rev30"])
        .map(lambda v: f"color:{C_MAUVAIS}" if pd.notna(v) and v > 0.20 else "", subset=["croiss_actif"])
        .map(lambda v: f"color:{C_MAUVAIS}" if pd.notna(v) and v > 0.02 else (f"color:{C_BON}" if pd.notna(v) and v < -0.005 else ""), subset=["emission"])
        .map(lambda v: f"color:{C_BON};font-weight:600" if isinstance(v, str) and v.startswith("↑") else (f"color:{C_MAUVAIS};font-weight:600" if isinstance(v, str) and v.startswith("↓") else ""), subset=["derive"])
        .apply(lambda col: [f"color:{C_MAUVAIS};font-weight:600" if m_ >= 2 else (f"color:{C_BON};font-weight:600" if p_ >= 2 and m_ == 0 else "")
                            for p_, m_ in zip(vue["plus"], vue["moins"])], subset=["bilan"]),
        column_config={"nom": "Ligne", "surprise_txt": "Surprise", "derive": "Dérive",
                       "rev90": "BPA 90 j", "net_rev30": "Analystes",
                       "gp_a": "Rentab.", "croiss_actif": "Actif 1 an", "emission": "Émissions",
                       "bilan": "Signaux"})
    st.caption("Surprise : bénéfice publié face au consensus. Dérive : sens attendu du cours après la surprise et jours restants. "
               "BPA 90 j : révision du bénéfice attendu pour l'exercice en cours. Analystes : nombre de hausses moins nombre de baisses de prévision sur 30 jours. Rentab. : rentabilité brute. "
               "Actif 1 an : croissance de l'actif total. Émissions : variation du nombre d'actions (négatif = rachats).")
    st.caption("« — » : donnée absente chez Yahoo (fréquent pour les petites valeurs et les financières). La rentabilité brute n'a pas de sens pour les banques et assureurs.")

    st.markdown("### Croissance, cash et croisement prix contre bénéfices")
    intro("Le momentum de prix tient mieux quand les bénéfices suivent (Chan, Jegadeesh et Lakonishok, 1996 ; Novy-Marx, 2015). "
          "Ce tableau croise la tendance du cours avec la révision des bénéfices attendus, et ajoute la croissance publiée et le rendement du free cash flow.")
    pu_all = None
    def _mom(t):
        nonlocal pu_all
        if t in pf.index and pd.notna(pf.loc[t].get("momentum")):
            return float(pf.loc[t, "momentum"])
        try:
            if pu_all is None:
                pu_all, _, _ = prix_univers(False)
            x = pu_all[t].dropna() if t in pu_all.columns else pd.Series(dtype=float)
            return float(x.iloc[-22] / x.iloc[-253] - 1) if len(x) > 260 else np.nan
        except Exception:
            return np.nan
    def _croise(m, r):
        if pd.isna(m) or pd.isna(r):
            return "—"
        if m > 0 and r > 0.01:
            return "Tendance confirmée"
        if m > 0 and r < -0.01:
            return "Hausse sans soutien"
        if m <= 0 and r > 0.01:
            return "À étudier"
        if m <= 0 and r < -0.01:
            return "À éviter"
        return "Neutre"
    cr = pd.DataFrame({"nom": df["nom"].values,
                       "croiss_ca": [pd.to_numeric(res.loc[t].get("croiss_ca"), errors="coerce") if t in res.index else np.nan for t in df["ticker"]],
                       "croiss_bpa": [pd.to_numeric(res.loc[t].get("croiss_bpa"), errors="coerce") if t in res.index else np.nan for t in df["ticker"]],
                       "fcf_rdt": [pd.to_numeric(res.loc[t].get("fcf_rdt"), errors="coerce") if t in res.index else np.nan for t in df["ticker"]],
                       "momentum": [_mom(t) for t in df["ticker"]], "rev90": df["rev90"].values})
    cr["croisement"] = [_croise(m, r) for m, r in zip(cr["momentum"], cr["rev90"])]
    ordre_c = {"Hausse sans soutien": 0, "À éviter": 1, "À étudier": 2, "Tendance confirmée": 3, "Neutre": 4, "—": 5}
    cr = cr.iloc[np.argsort([ordre_c[v] for v in cr["croisement"]], kind="stable")]
    coul_c = {"Tendance confirmée": C_BON, "Hausse sans soutien": C_MAUVAIS, "À éviter": C_MAUVAIS, "À étudier": PALETTE[0]}
    tableau(cr.style.format({"croiss_ca": f_pct(0), "croiss_bpa": f_pct(0), "fcf_rdt": f_pct(1, False), "momentum": f_pct(0), "rev90": f_pct(1)})
            .map(signe_couleur, subset=["croiss_ca", "croiss_bpa", "momentum", "rev90"])
            .map(lambda v: f"color:{coul_c[v]};font-weight:600" if v in coul_c else "", subset=["croisement"]),
            hide_index=True, column_config={"nom": "Ligne", "croiss_ca": "Chiffre d'affaires 1 an", "croiss_bpa": "BPA 1 an", "fcf_rdt": "Rendement du FCF",
                                            "momentum": "Momentum 12-1", "rev90": "BPA attendu 90 j", "croisement": "Croisement"})
    st.caption("Tendance confirmée : le cours monte et les bénéfices attendus sont relevés. Hausse sans soutien : le cours monte mais les bénéfices sont revus en baisse, "
               "le cas où le momentum risque le plus de se retourner. À étudier : le cours baisse alors que les bénéfices sont relevés, le marché n'a peut-être pas encore suivi. "
               "À éviter : les deux baissent. Rendement du FCF : free cash flow du dernier exercice ÷ capitalisation ; il n'a pas de sens pour les banques et les assureurs.")

    g, d = st.columns(2)
    with g:
        z = df.dropna(subset=["rev90"]).sort_values("rev90")
        if len(z):
            fig = go.Figure(go.Bar(x=z["rev90"], y=z["nom"], orientation="h", marker_color=[C_BON if v >= 0 else C_MAUVAIS for v in z["rev90"]],
                                   text=[pct(v, 1, True) for v in z["rev90"]], textposition="outside", hovertemplate="%{y} : %{x:+.1%}<extra></extra>"))
            lo, hi = min(0.0, float(z["rev90"].min())), max(0.0, float(z["rev90"].max()))
            pad = 0.45 * max(hi - lo, 0.01)
            fig.update_layout(height=max(280, 22 * len(z)), margin=dict(t=36, l=0, r=10, b=0), xaxis_tickformat="+.0%",
                              xaxis_range=[lo - pad * (lo < 0), hi + pad * (hi > 0)],
                              title=dict(text="Révision du bénéfice attendu sur 90 jours", font_size=13))
            graphe(fig, key="rev90")
    with d:
        z = df.dropna(subset=["gp_a", "croiss_actif"])
        if len(z):
            fig = go.Figure(go.Scatter(x=z["croiss_actif"], y=z["gp_a"], mode="markers+text", text=z["nom"].str.replace(" (candidat)", "*", regex=False),
                                       textposition="top center", textfont_size=10, marker=dict(size=10, color=[PALETTE[0] if x else PALETTE[2] for x in z["detenu"]]),
                                       hovertemplate="%{text}<br>Croissance de l'actif %{x:+.0%}<br>Rentabilité brute %{y:.0%}<extra></extra>"))
            fig.add_hline(y=0.30, line_dash="dot", line_color=C_NEUTRE)
            fig.add_vline(x=0.20, line_dash="dot", line_color=C_NEUTRE)
            fig.update_layout(height=max(280, 22 * len(df.dropna(subset=['rev90']))), margin=dict(t=36, l=0, r=30, b=0), xaxis_tickformat="+.0%", yaxis_tickformat=".0%",
                              xaxis_title="Croissance de l'actif sur un an", yaxis_title="Rentabilité brute",
                              title=dict(text="Qualité et discipline : en haut à gauche, c'est mieux", font_size=13))
            graphe(fig, key="qualite")

    st.markdown("### Comment lire ces indicateurs")
    st.markdown(
        "- **Surprise de résultats et dérive post-annonce** (Bernard et Thomas, 1989). L'écart entre le bénéfice publié et le consensus. "
        "Le cours tend à continuer dans le sens de la surprise pendant environ 60 jours de bourse (≈ 3 mois) : c'est l'une des anomalies les plus robustes. "
        "Seuil retenu : ±2 %.\n"
        "- **Révisions des analystes** (Chan, Jegadeesh et Lakonishok, 1996). Les titres dont le bénéfice attendu est revu à la hausse continuent de "
        "surperformer, car les analystes ajustent lentement. « Analystes ↑ − ↓ » compte ceux qui ont relevé moins ceux qui ont abaissé sur 30 jours.\n"
        "- **Rentabilité brute** (Novy-Marx, 2013). Marge brute ÷ actif total. Plus prédictive que le ROE, qui peut être gonflé par la dette. "
        "Au-dessus de 30 %, c'est une entreprise de qualité ; sous 10 %, c'est faible (sauf modèles très capitalistiques comme les utilities).\n"
        "- **Croissance de l'actif** (Cooper, Gulen et Schill, 2008). Les entreprises qui grossissent très vite (acquisitions, investissements massifs) "
        "font en moyenne moins bien ensuite. Alerte au-dessus de +20 % sur un an.\n"
        "- **Émission d'actions** (Pontiff et Woodgate, 2008). Une entreprise qui émet des actions les juge souvent chères ; une entreprise qui en rachète, "
        "bon marché. Positif sous −0,5 % (rachats), négatif au-dessus de +2 %.\n\n"
        "**Signaux + / −** : le nombre d'indicateurs favorables et défavorables. Deux signaux négatifs ou plus sur une ligne détenue méritent de relire sa fiche. "
        "Ces effets sont mesurés sur des milliers d'actions : sur une ligne isolée, ils déplacent les probabilités, ils ne garantissent rien.")


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

    st.markdown("### Ton portefeuille face au marché : active share")
    intro("La part de ton portefeuille qui diffère du marché (Cremers et Petajisto, 2009) : 0 % = une copie de l'indice, 100 % = aucun titre en commun. "
          "C'est la réponse chiffrée à « vos actions sont toutes dans l'indice » : ce qui compte, ce sont les poids, pas la liste des titres. "
          "Le marché est approché par ton univers large et tes lignes, pondérés par leur capitalisation.")
    if st.button("Calculer l'active share (environ une minute la première fois)", key="as_btn"):
        st.session_state["as_charge"] = True
    if st.session_state.get("as_charge"):
        u_ = charger_univers()
        t_idx = tuple(sorted(set(u_.index) | set(pf.index)))
        caps = capitalisations(t_idx).dropna()
        w_p = pf["valeur"].dropna()
        a_s = A.active_share(w_p, caps)
        commun = caps.index.intersection(w_p.index)
        c = st.columns(3)
        kpi(c[0], "Active share", pct(a_s, 0) if pd.notna(a_s) else "—", "au-dessus de 80 % : gestion vraiment active", delta_color="off")
        kpi(c[1], "Recouvrement avec le marché", pct(1 - a_s, 0) if pd.notna(a_s) else "—", f"{len(commun)} titres en commun sur {len(caps)}", delta_color="off")
        kpi(c[2], "Poids de tes titres dans le marché", pct(caps.reindex(commun).sum() / caps.sum(), 0) if len(caps) else "—",
            "contre 100 % de ton portefeuille", delta_color="off")
        st.caption("Cremers et Petajisto classent les fonds au-dessus de 80 % parmi les vrais gérants actifs, et ceux sous 60 % parmi les « faux actifs » "
                   "qui facturent une gestion active pour un quasi-indice. Un active share élevé ne prouve aucun talent : il dit seulement que ton résultat "
                   "pourra beaucoup s'écarter de celui du marché, en bien comme en mal.")

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
- **Le « i »** à côté d'un chiffre ou d'un titre de colonne : passe la souris dessus (ou touche-le sur téléphone) pour voir ce que c'est, comment le lire et un exemple chiffré.
- **Laboratoire** (en bas du menu) : les pistes testées qui ne changent pas une décision au quotidien (portefeuille modèle, optimisation, momentum, régime, scénarios politiques, contrôle des données).
- Les mêmes parcours sont cliquables en bas du *Tableau de bord* (« Par où passer »), et chaque page propose en bas « Et ensuite ? ».
- **Chaque jour, 2 minutes** : le *Tableau de bord*. Regarde les alertes rouges et les contraintes, puis la carte des lignes.
- **Avant chaque achat** : *Candidats* pour trouver, *Valorisation*, *Résultats et qualité* et MarketScreener pour vérifier, *Simulateur* pour tester, puis écris la thèse, la menace et la condition de sortie dans lignes.csv.
- **Le 1er de chaque mois** : *Rééquilibrage* (ordres et test de la page blanche), *Règles*, *Risque*, *Stress tests*, puis *Chance ou talent* sur le mois écoulé. Note trois chiffres : ton percentile à risque égal, ton ratio de diversification, ta plus forte baisse. Termine par le *Rapport*, à enregistrer en PDF.
- **Après chaque ordre** : ajoute la ligne dans transactions.csv (date AAAA-MM-JJ, point décimal, prix en euros = montant NextWise ÷ quantité).
""",
    "Score du concours": """
- Les quatre critères sans chance : **HHI** (plus bas = moins concentré ; le minimum avec N lignes à poids égaux est 1/N), **secteurs couverts sur 11**, **ratio de diversification**, **UCITS**. Plus les **quiz**.
- La liste « ce qui ferait gagner des points » est recalculée à chaque ouverture : secteur manquant, écart à la diversification maximale, cash qui ne travaille pas, contrainte dépassée.
- Le message tout prêt pour les organisateurs : pondération des critères, mesure du risque, dividendes.
""",
    "Résultats et qualité": """
- **À ouvrir pendant chaque saison de résultats** (fin janvier – mars, juillet, octobre) et avant chaque achat. Clique sur le bouton une fois : les données sont gardées 24 h et alimentent les alertes du tableau de bord.
- **Dérive post-annonce** : après une surprise de plus de 2 %, le cours tend à continuer dans le même sens pendant environ 3 mois. Une flèche rouge sur une ligne détenue = relire sa fiche ; une flèche verte sur un candidat = bon moment pour l'étudier.
- **Révisions** : un bénéfice attendu abaissé de plus de 5 % en 90 jours déclenche une alerte orange.
- **Rentabilité brute et croissance de l'actif** : en haut à gauche du nuage (rentable et discipliné), c'est le profil qui fait le mieux en moyenne.
- Teste n'importe quel candidat avec le champ des tickers libres, comme dans Rééquilibrage.
""",
    "Portefeuille modèle": """
- **À quoi ça sert** : obtenir une proposition de portefeuille purement quantitative (signaux robustes + poids fondés sur le risque + limites du concours), puis vérifier par une batterie de tests si elle aurait marché hors échantillon.
- **Comment l'utiliser** : garde les réglages par défaut, regarde la section 2, puis lance les confirmations. Ne commence à étudier les titres proposés que si presque tous les tests sont confirmés.
- **Lecture des tests** : « Hasard » compare à des portefeuilles tirés au sort aux mêmes dates ; « Alpha » retire l'effet du marché ; « Sharpe dégonflé » pénalise le nombre de réglages essayés ; « Stabilité » vérifie que ça ne tient pas à une seule année ; « Robustesse » vérifie que les réglages voisins marchent aussi.
- **Ne bouge pas les curseurs jusqu'à ce que ça passe** : chaque essai est compté et rend le test plus exigeant.
- **Ensuite** : colle les titres proposés dans « Résultats et qualité », puis fais ta fiche (thèse, menace, sortie) pour ceux qui t'intéressent.
""",
    "Scénarios politiques": """
- **À ouvrir** quand la campagne présidentielle française agite les marchés, ou quand tes lignes françaises baissent sans raison apparente.
- **Section 1** : les probabilités d'un marché de prédiction (ou saisies à la main). Tu choisis toi-même les issues qui inquiéteraient les marchés.
- **Section 2** : si la statistique t dépasse 2,6 en valeur absolue, la politique explique une partie de l'écart entre le CAC 40 et l'Europe. Sinon, cherche la cause ailleurs.
- **Section 3** : seules les sensibilités significatives comptent. Ce sont les lignes à alléger si tu veux réduire ce risque.
- **Section 5** : ce que tu gagnes ou perds si la probabilité bouge. Ce n'est pas une prévision : pondéré par les probabilités, le résultat attendu est nul.
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
# Le menu : seules les pages qui mènent à une décision ou servent devant le jury sont en haut.
# « Laboratoire » regroupe les pistes testées mais qui ne changent pas une décision au quotidien : elles restent consultables.
NAV = {
    "Suivi": [
        st.Page(page_direct, title="Tableau de bord", icon=":material/dashboard:", default=True),
        st.Page(page_score, title="Score du concours", icon=":material/emoji_events:", url_path="score"),
        st.Page(page_lignes, title="Lignes", icon=":material/list_alt:", url_path="lignes"),
        st.Page(page_regles, title="Règles", icon=":material/gavel:", url_path="regles"),
        st.Page(page_journal, title="Journal", icon=":material/receipt_long:", url_path="journal"),
    ],
    "Décider": [
        st.Page(page_modele, title="Portefeuille modèle", icon=":material/auto_awesome:", url_path="modele"),
        st.Page(page_candidats, title="Candidats", icon=":material/travel_explore:", url_path="candidats"),
        st.Page(page_valorisation, title="Valorisation", icon=":material/sell:", url_path="valorisation"),
        st.Page(page_resultats, title="Résultats et qualité", icon=":material/fact_check:", url_path="resultats"),
        st.Page(page_simulateur, title="Simulateur", icon=":material/science:", url_path="simulateur"),
        st.Page(page_reequilibrage, title="Rééquilibrage", icon=":material/balance:", url_path="reequilibrage"),
    ],
    "Risque": [
        st.Page(page_risque, title="Risque", icon=":material/shield:", url_path="risque"),
        st.Page(page_stress, title="Stress tests", icon=":material/thunderstorm:", url_path="stress-tests"),
    ],
    "Jury": [
        st.Page(page_performance, title="Performance", icon=":material/trending_up:", url_path="performance"),
        st.Page(page_chance, title="Chance ou talent", icon=":material/casino:", url_path="chance-ou-talent"),
        st.Page(page_rapport, title="Rapport", icon=":material/summarize:", url_path="rapport"),
        st.Page(page_guide, title="Guide", icon=":material/menu_book:", url_path="guide"),
    ],
    "Laboratoire": [
        st.Page(page_optimisation, title="Optimisation", icon=":material/tune:", url_path="optimisation"),
        st.Page(page_momentum, title="Momentum", icon=":material/speed:", url_path="momentum"),
        st.Page(page_regime, title="Régime de marché", icon=":material/waves:", url_path="regime"),
        st.Page(page_politique, title="Scénarios politiques", icon=":material/how_to_vote:", url_path="politique"),
        st.Page(page_donnees, title="Données", icon=":material/database:", url_path="donnees"),
    ],
}
PAGES = {(pg_.url_path or ""): pg_ for groupe_ in NAV.values() for pg_ in groupe_}

# Les parcours : dans quel ordre ouvrir les pages selon ce qu'on vient faire (affichés sur le tableau de bord).
PARCOURS = {
    "Chaque jour, 2 minutes": [("", "les alertes et la carte des lignes"), ("lignes", "la fiche du titre cité par une alerte")],
    "Avant un ordre": [("candidats", "trouver une idée qui respecte tes règles"), ("resultats", "vérifier que les bénéfices suivent"),
                       ("valorisation", "vérifier le prix payé"), ("simulateur", "tester l'ordre avant de le passer")],
    "Le 1er du mois": [("reequilibrage", "les ordres pour revenir aux poids cibles"), ("regles", "les règles encore respectées ?"),
                       ("risque", "volatilité, bêta, contributions"), ("stress-tests", "ce que ferait une crise"), ("rapport", "le bilan à garder en PDF")],
    "Pour le jury": [("score", "où tu en es sur les critères"), ("chance-ou-talent", "ce qui relève du hasard"),
                     ("performance", "d'où vient le résultat"), ("rapport", "le document à montrer")],
}

# La suite logique : ce qu'on ouvre en général après chaque page (affichée en bas de page).
SUITE = {
    "score": [("chance-ou-talent", "le résultat tient-il au hasard ?"), ("performance", "d'où vient le résultat")],
    "lignes": [("resultats", "les bénéfices du titre"), ("valorisation", "le prix payé"), ("simulateur", "tester une vente")],
    "regles": [("reequilibrage", "corriger les écarts"), ("candidats", "remplacer une ligne")],
    "journal": [("performance", "ce que les ordres ont rapporté")],
    "candidats": [("resultats", "vérifier les bénéfices"), ("valorisation", "vérifier le prix"), ("simulateur", "tester l'achat")],
    "modele": [("candidats", "étudier les titres proposés"), ("simulateur", "tester un changement")],
    "optimisation": [("simulateur", "tester les poids proposés"), ("reequilibrage", "en faire des ordres")],
    "simulateur": [("reequilibrage", "chiffrer les ordres"), ("journal", "enregistrer l'ordre passé")],
    "reequilibrage": [("simulateur", "tester avant de passer l'ordre"), ("journal", "enregistrer l'ordre passé")],
    "risque": [("stress-tests", "ce que ferait une crise"), ("reequilibrage", "alléger ce qui pèse trop")],
    "stress-tests": [("risque", "ton risque aujourd'hui"), ("simulateur", "tester un allègement")],
    "politique": [("lignes", "la fiche des lignes exposées"), ("simulateur", "tester un allègement")],
    "regime": [("risque", "ton risque aujourd'hui"), ("stress-tests", "ce que ferait une crise")],
    "performance": [("chance-ou-talent", "chance ou talent ?"), ("rapport", "garder une trace")],
    "chance-ou-talent": [("score", "ce que ça change pour le concours"), ("rapport", "garder une trace")],
    "momentum": [("resultats", "la hausse est-elle soutenue par les bénéfices ?")],
    "valorisation": [("resultats", "croissance et qualité des bénéfices")],
    "resultats": [("valorisation", "le prix payé"), ("simulateur", "tester un ordre")],
    "rapport": [("guide", "comment lire chaque page")],
}


def suite_logique(u):
    etapes = [(x, q) for x, q in SUITE.get(u, []) if x in PAGES]
    if not etapes:
        return
    st.html('<div style="margin-top:28px;border-top:1px solid rgba(128,128,128,.25);padding-top:10px;font-size:.78rem;letter-spacing:.06em;'
            'text-transform:uppercase;opacity:.65">Et ensuite ?</div>')
    for col, (x, quoi) in zip(st.columns(3), etapes):
        col.page_link(PAGES[x], label=f"{PAGES[x].title} : {quoi}")


pg = st.navigation(NAV, expanded=True)
st.session_state["page"] = pg.title
bandeau()
pg.run()
suite_logique(pg.url_path or "")
