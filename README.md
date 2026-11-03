# NextWise · suivi en direct

Appli Streamlit qui suit le portefeuille NextWise à partir d'un journal de transactions.
Cours Yahoo Finance différés d'environ 15 minutes, convertis en euros. Les chiffres officiels restent ceux de NextWise.

## Fichiers
- `app.py` : l'appli
- `transactions.csv` : le journal (une ligne par ordre exécuté)
- `lignes.csv` : la fiche de chaque ligne (poche, secteur, place, thèse, menace, sortie)
- `univers.csv` : les titres utilisés pour tirer les portefeuilles aléatoires (onglet « Chance ou talent »)
- `requirements.txt` : les bibliothèques

## Types de transaction
`Apport`, `Achat`, `Vente`, `Frais`, `Dividende`, `Retrait`. Prix en euros, comme sur NextWise.
Tickers au format Yahoo : `KBC.BR`, `AI.PA`, `RACE.MI`, `KPN.AS`, `EDP.LS`, `RYA.IR`, `EQNR.OL`…

## Google Sheet (optionnel, pour modifier depuis le téléphone)
1. Crée un Google Sheet avec deux onglets nommés exactement `transactions` et `lignes`.
2. Importe `transactions.csv` dans le premier et `lignes.csv` dans le second (Fichier › Importer › Remplacer la feuille actuelle).
3. Partager › Tous les utilisateurs disposant du lien › Lecteur.
4. Copie l'identifiant du Sheet (la partie entre `/d/` et `/edit` dans l'adresse).
5. Sur Streamlit Cloud : Settings › Secrets, colle `SHEET_ID = "ton_identifiant"`.

## Tester sans connexion
`NEXTWISE_MOCK=1 streamlit run app.py` (cours simulés).
