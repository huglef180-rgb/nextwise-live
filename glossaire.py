# -*- coding: utf-8 -*-
"""Le petit « i » du site : pour chaque indicateur, ce que c'est, comment le lire, et un exemple chiffré.
Une entrée = (c'est quoi, comment le lire, exemple). Les libellés du site sont reliés aux entrées par aide()."""
import re
import unicodedata

G = {
    # ------------------------------------------------------------------ risque
    "volatilite prevue": (
        "L'ampleur des variations attendues du portefeuille sur un an, estimée à partir des cours récents (les jours récents comptent plus que les anciens).",
        "Plus le chiffre est bas, plus le parcours est calme. Environ deux années sur trois, le résultat tombe à plus ou moins une volatilité de la moyenne. "
        "Un indice d'actions européen tourne autour de 15 %.",
        "Avec 10 % de volatilité sur 50 000 €, une année ordinaire se termine entre −5 000 € et +5 000 € autour du rendement moyen. Une journée ordinaire bouge de 10 % ÷ 16 ≈ 0,6 %, soit environ 300 €."),
    "volatilite": (
        "L'ampleur habituelle des variations d'un titre ou d'un portefeuille, ramenée à une année. C'est l'écart-type des rendements quotidiens multiplié par 16 (racine de 252 jours de bourse).",
        "Ce n'est pas la perte maximale : c'est la taille d'un mouvement « normal ». En dessous de 15 % : calme. De 15 à 25 % : action classique. Au-dessus de 35 % : titre nerveux.",
        "Une action à 32 % de volatilité bouge en moyenne de 32 ÷ 16 = 2 % par jour. Une autre à 16 % bouge de 1 % par jour : à poids égal, la première pèse deux fois plus dans ton risque."),
    "beta": (
        "La sensibilité au marché : de combien le titre (ou le portefeuille) bouge en moyenne quand l'indice bouge de 1 %.",
        "1 = comme le marché. Moins de 1 = amortit les mouvements. Plus de 1 = les amplifie. Un bêta bas protège dans les baisses mais fait prendre du retard dans les fortes hausses.",
        "Avec un bêta de 0,67, si l'Euronext 100 perd 3 %, ton portefeuille perd en moyenne 0,67 × 3 = 2 %. S'il gagne 3 %, tu gagnes environ 2 %."),
    "ratio de diversification": (
        "La moyenne des volatilités de tes lignes divisée par la volatilité du portefeuille entier (Choueifaty et Coignard, 2008). Il mesure ce que le mélange des lignes fait gagner en calme.",
        "1 = aucune diversification (tout bouge ensemble). Plus c'est haut, mieux c'est. Au-dessus de 2 : très bien diversifié. Son carré donne le nombre de paris réellement indépendants.",
        "Tes lignes ont en moyenne 26 % de volatilité et ton portefeuille 10,3 % : ratio = 26 ÷ 10,3 ≈ 2,5. Au carré : 6,4. Tes 26 lignes se comportent donc comme environ 6 paris indépendants."),
    "paris independants": (
        "Le carré du ratio de diversification : le nombre de paris sans lien entre eux auquel ton portefeuille équivaut vraiment.",
        "Toujours bien plus bas que le nombre de lignes, parce que les actions montent et baissent souvent ensemble. Au-dessus de 5 pour un portefeuille d'actions, c'est bon.",
        "26 lignes, ratio de diversification 2,53 : 2,53² = 6,4 paris. Ajouter une 27e banque n'en ajoute presque aucun ; ajouter de l'or en ajoute, car il ne suit pas les actions."),
    "hhi": (
        "L'indice de concentration de Herfindahl : la somme des carrés des poids. Il dit si l'argent est réparti ou entassé sur quelques lignes.",
        "Plus c'est bas, mieux c'est réparti. Le minimum est 1 ÷ nombre de lignes. 1 ÷ HHI donne le « nombre effectif de lignes ».",
        "26 lignes à poids égaux : HHI = 1 ÷ 26 = 3,8 %. Si une ligne monte à 20 % du portefeuille, le HHI grimpe vers 7 % : tu n'as plus que 1 ÷ 0,07 ≈ 14 lignes effectives."),
    "nombre effectif de lignes": (
        "1 divisé par le HHI : le nombre de lignes à poids égaux qui donnerait la même concentration que ton portefeuille.",
        "Proche de ton nombre réel de lignes = poids bien répartis. Nettement plus bas = quelques lignes dominent.",
        "Tu as 26 lignes et un HHI de 4,0 % : 1 ÷ 0,040 = 25 lignes effectives. Presque aucune perte : tes poids sont quasi égaux."),
    "var 95": (
        "La « valeur en risque » : la perte sur une journée qui n'est dépassée qu'une fois sur vingt (5 % des jours).",
        "C'est un seuil, pas un maximum : un jour sur vingt, la perte est pire. La version Cornish-Fisher corrige le fait que les krachs sont plus fréquents que ne le dit la loi normale.",
        "Une VaR de 1,1 % sur 50 000 € = 550 €. Environ une fois par mois de bourse (1 jour sur 20), tu perdras plus de 550 € dans la journée."),
    "perte moyenne au-dela": (
        "La perte moyenne des 5 % de journées les plus mauvaises (en anglais : expected shortfall). Elle répond à « quand ça dépasse la VaR, ça donne quoi ? ».",
        "Toujours pire que la VaR. Un grand écart entre les deux signale des mauvais jours vraiment violents.",
        "VaR à 1,1 % et perte moyenne au-delà à 1,6 % : dans tes pires journées, tu perds en moyenne 800 € sur 50 000 €, pas 550 €."),
    "plus forte baisse": (
        "La plus grande chute entre un sommet et le creux qui a suivi (en anglais : drawdown maximal).",
        "C'est la perte qu'aurait subie quelqu'un entré au pire moment. Plus c'est proche de 0, mieux c'est. C'est la mesure de risque la plus parlante pour un jury.",
        "Le portefeuille monte à 52 000 €, redescend à 49 400 €, puis remonte : plus forte baisse = (49 400 − 52 000) ÷ 52 000 = −5 %."),
    "part du risque": (
        "La part de la volatilité totale du portefeuille qui vient de cette ligne. Elle dépend de son poids, de sa nervosité et de sa ressemblance avec les autres lignes.",
        "À comparer au poids. Part du risque supérieure au poids : la ligne prend plus de place dans ton risque que dans ton argent.",
        "Une ligne pèse 4 % du portefeuille mais porte 8 % du risque : elle compte double. C'est la première à alléger si tu veux calmer le portefeuille."),
    "delta volatilite": (
        "Ce que deviendrait la volatilité du portefeuille si tu ajoutais ce titre (ou passais cet ordre).",
        "Négatif = l'ordre calme le portefeuille. Positif = il l'agite.",
        "Δ volatilité de −0,3 point : ton portefeuille passerait de 10,3 % à 10,0 % de volatilité."),
    "delta diversification": (
        "Ce que deviendrait le ratio de diversification si tu ajoutais ce titre.",
        "Positif = le titre apporte un pari nouveau. Proche de zéro ou négatif = il ressemble à ce que tu as déjà.",
        "Δ diversification de +0,05 : ton ratio passerait de 2,53 à 2,58."),
    "correlation": (
        "À quel point deux séries bougent ensemble, de −1 (toujours en sens inverse) à +1 (toujours ensemble).",
        "Sous 0,3 : le titre vit sa vie, il diversifie vraiment. Au-dessus de 0,7 : il fait presque doublon avec ce que tu as déjà.",
        "Deux banques de la zone euro : corrélation d'environ 0,8, la seconde n'apporte presque rien. L'or face aux actions : proche de 0, il amortit."),
    "ucits 5/10/40": (
        "La règle des fonds européens : aucune ligne au-dessus de 10 %, et les lignes de plus de 5 % ne doivent pas dépasser 40 % du portefeuille à elles toutes.",
        "« Respectée » = ton portefeuille pourrait être celui d'un vrai fonds réglementé. C'est un argument simple devant un jury.",
        "Trois lignes à 6 % et le reste sous 5 % : les « grosses » pèsent 18 %, bien sous 40 %. La règle est respectée."),
    "jours pour tout vendre": (
        "Le nombre de séances nécessaires pour vendre toute la ligne sans dépasser 10 % du volume échangé chaque jour.",
        "Moins de 1 jour : aucune difficulté. Plusieurs jours : en vrai, ta vente ferait baisser le cours.",
        "Tu détiens 2 000 € d'un titre qui échange 5 millions d'euros par jour : 10 % du volume = 500 000 €, tu sors en une fraction de séance."),
    "turbulence": (
        "L'indice de turbulence (Kritzman et Li, 2010) : à quel point les mouvements récents des secteurs sont inhabituels, par leur taille et par leurs liens entre eux.",
        "Sous le seuil (95e centile de l'historique) : marché normal. Au-dessus : les repères habituels ne tiennent plus, les corrélations montent et la diversification protège moins.",
        "Turbulence à 12 pour un seuil à 25 : rien d'anormal. En mars 2020, l'indice dépassait largement le seuil pendant des semaines."),
    "ratio d'absorption": (
        "La part des mouvements de tous les secteurs expliquée par quelques facteurs communs (Kritzman et al., 2011).",
        "Haut = tout bouge ensemble, le marché est fragile : un choc se propage partout. Bas = les secteurs vivent leur vie.",
        "Ratio à 85 % : un seul choc peut entraîner tout le marché. À 60 % : une mauvaise nouvelle sur les banques ne touche pas forcément la santé."),
    "exposition conseillee": (
        "La part à garder investie dans la stratégie momentum pour viser un risque constant (Barroso et Santa-Clara, 2015) : volatilité cible divisée par volatilité récente du facteur.",
        "Sous 1 : le momentum est agité, il faut réduire. Au-dessus de 1 : il est calme. C'est la parade documentée contre les krachs du momentum.",
        "Cible de 12 % et volatilité récente du facteur de 20 % : 12 ÷ 20 = 0,6. Il faudrait ne garder que 60 % de l'exposition habituelle."),
    # ------------------------------------------------------------------ performance
    "ratio de sharpe": (
        "Le rendement gagné au-dessus du taux sans risque, divisé par la volatilité. C'est le rendement par unité de risque.",
        "Sous 0 : tu aurais mieux fait de rester en dépôt. Autour de 0,5 : dans la moyenne des actions sur longue période. Au-dessus de 1 : très bon, mais rare sur la durée. "
        "Sur quelques semaines, il ne veut presque rien dire.",
        "Rendement de 9 % par an, taux sans risque de 2,5 %, volatilité de 13 % : (9 − 2,5) ÷ 13 = 0,5."),
    "probabilite que le vrai sharpe soit positif": (
        "La probabilité que ton Sharpe « vrai » soit au-dessus de zéro, compte tenu du peu de données et de la forme des rendements (Bailey et López de Prado, 2012).",
        "50 % = pile ou face. Il faut dépasser 95 % pour parler de preuve. Avec quelques semaines d'historique, c'est presque toujours loin du compte.",
        "Sharpe de 1,2 sur 25 séances : la probabilité qu'il soit vraiment positif tourne autour de 65 %. Une chance sur trois que ce soit du bruit."),
    "sharpe a battre vu le nombre d'essais": (
        "Le Sharpe que le hasard seul produit quand on essaie beaucoup de réglages et qu'on garde le meilleur (Sharpe « dégonflé », Bailey et López de Prado, 2014).",
        "Si le Sharpe de ton modèle est en dessous, ton résultat peut n'être que le meilleur tirage parmi tes essais. Plus tu essaies de réglages, plus la barre monte.",
        "Tu as testé 13 réglages : le meilleur de 13 modèles sans aucun talent affiche déjà un Sharpe d'environ 0,6. Un modèle à 0,5 ne prouve donc rien."),
    "alpha": (
        "Le rendement qui reste une fois retiré ce que l'exposition au marché (et aux autres facteurs) explique. C'est la part attribuable à la sélection.",
        "Positif = tu as fait mieux que ce que ton niveau de risque justifiait. Mais il faut regarder son t (ou sa p-valeur) : un alpha non significatif est peut-être de la chance.",
        "Le marché fait +10 %, ton bêta est 0,7 : le risque pris justifie 7 %. Tu as fait +9 % : alpha = +2 %."),
    "statistique t": (
        "L'estimation divisée par son incertitude. Elle dit si un chiffre se distingue de zéro ou s'il est noyé dans le bruit.",
        "Au-delà de 2 en valeur absolue : significatif au seuil habituel de 5 %. Quand on a testé beaucoup d'idées, la recherche exige plutôt 3 (Harvey, Liu et Zhu, 2016).",
        "Alpha de 2 % avec une incertitude de 4 % : t = 0,5, rien de prouvé. Le même alpha avec une incertitude de 0,8 % : t = 2,5, significatif."),
    "p-valeur": (
        "La probabilité d'observer un résultat au moins aussi bon si, en réalité, tu n'avais aucun talent (le hasard seul).",
        "Petite = le hasard explique mal ton résultat. Sous 5 % : significatif. Attention : ce n'est pas la probabilité d'avoir du talent, et une p-valeur basse ne dit rien de la taille de l'effet.",
        "p-valeur de 30 % : sur 100 gérants sans aucun talent, 30 auraient fait aussi bien que toi. Impossible de conclure. À 2 % : seulement 2 sur 100."),
    "r2": (
        "La part des variations expliquée par le modèle (le marché, les facteurs…), de 0 à 100 %.",
        "Haut = ton portefeuille suit de près ce qui l'explique, il reste peu de place pour la sélection. Bas = il vit sa vie.",
        "R² de 70 % face à l'Euronext 100 : 70 % de tes mouvements quotidiens viennent du marché, 30 % de tes choix de titres."),
    "ecart de suivi": (
        "La volatilité de l'écart entre ton rendement et celui de l'indice (en anglais : tracking error).",
        "Sous 2 % : tu colles à l'indice. De 4 à 8 % : gestion vraiment active. Plus il est haut, plus tu peux t'écarter de l'indice, en bien comme en mal.",
        "Écart de suivi de 6 % : deux années sur trois, ton rendement finit entre 6 points sous l'indice et 6 points au-dessus."),
    "active share": (
        "La part du portefeuille qui diffère de l'indice : la moitié de la somme des écarts de poids, titre par titre (Cremers et Petajisto, 2009).",
        "0 % = copie de l'indice. Sous 60 % : indice déguisé. Au-dessus de 80 % : vraie sélection de titres.",
        "L'indice a 5 % d'ASML et toi 0 % ; toi 4 % de Kinepolis et l'indice 0 % : ces écarts s'additionnent. Avec 26 lignes face à 100 dans l'indice, tu es en général au-dessus de 80 %."),
    "effet allocation": (
        "Dans l'attribution de Brinson : ce que tu gagnes ou perds parce que tu pèses plus ou moins que l'indice sur chaque secteur.",
        "Positif = tes paris sectoriels ont payé (plus lourd sur les secteurs qui ont monté).",
        "Tu as 12 % d'énergie contre 6 % dans l'indice, et l'énergie fait 10 points de mieux que l'indice : effet allocation = 6 % × 10 = +0,6 point."),
    "effet selection": (
        "Dans l'attribution de Brinson : ce que tu gagnes ou perds parce que tes titres font mieux ou moins bien que leur secteur.",
        "Positif = tu as choisi les bons titres à l'intérieur des secteurs. C'est la preuve de sélection la plus directe.",
        "Tes banques font +8 %, les banques de l'indice +5 %, et les banques pèsent 20 % chez toi : effet sélection = 20 % × 3 = +0,6 point."),
    "asymetrie": (
        "La forme de la distribution des rendements : penche-t-elle vers les grosses pertes (négative) ou les gros gains (positive) ?",
        "Négative = petits gains fréquents et rares grosses pertes, le profil typique des actions et le plus dangereux. Proche de 0 = symétrique.",
        "Asymétrie de −0,8 : tes mauvaises journées sont plus violentes que tes bonnes. La volatilité seule sous-estime alors ton risque."),
    "kurtosis": (
        "L'épaisseur des « queues » : la fréquence des journées extrêmes par rapport à ce que prévoit la loi normale (qui vaut 3, ou 0 en excès).",
        "Au-dessus de la loi normale : les journées extrêmes arrivent plus souvent que prévu. C'est presque toujours le cas en bourse.",
        "Kurtosis en excès de 4 : une baisse « qui ne devrait arriver qu'une fois par siècle » selon la loi normale arrive en réalité tous les quelques ans."),
    "test de normalite": (
        "Un test qui demande : mes rendements suivent-ils une loi normale (la courbe en cloche) ?",
        "p-valeur sous 5 % : non, la loi normale est rejetée. Les calculs qui la supposent (VaR classique, p-valeur classique) sont alors trop optimistes : préfère le bootstrap et Cornish-Fisher.",
        "p = 0,1 % : tes rendements ont trop de journées extrêmes pour une loi normale. C'est le résultat habituel sur des actions."),
    "taux de reussite": (
        "La part des lignes (ou des paris) qui ont gagné.",
        "Ne dit rien tout seul : on peut gagner 9 fois sur 10 et perdre de l'argent si la dixième perte est énorme. À lire avec le gain moyen et la perte moyenne.",
        "18 lignes en gain sur 26 = 69 %. Mais si les 8 perdantes perdent chacune trois fois ce que gagne une gagnante, le total est négatif."),
    "pgr": (
        "PGR : part des gains que tu as encaissés parmi tous tes gains possibles. PLR : même chose pour les pertes (Odean, 1998).",
        "PGR nettement au-dessus de PLR = effet de disposition : tu vends tes gagnants trop tôt et tu gardes tes perdants. C'est le biais le plus répandu chez les investisseurs.",
        "Tu as vendu 3 de tes 10 gagnants (PGR = 30 %) et 0 de tes 6 perdants (PLR = 0 %) : écart de 30 points, le profil typique du biais."),
    "score de brier": (
        "La qualité de tes prévisions chiffrées : la moyenne des carrés des écarts entre la probabilité que tu as annoncée et ce qui est arrivé (0 ou 1).",
        "0 = parfait. 0,25 = aussi bien que de toujours dire 50 %. Au-dessus de 0,25 : pire que le hasard, souvent par excès de confiance.",
        "Tu annonces 80 % et ça arrive : (0,8 − 1)² = 0,04. Tu annonces 80 % et ça n'arrive pas : (0,8 − 0)² = 0,64. Une seule certitude ratée coûte très cher."),
    "performance sur risque": (
        "Ta performance divisée par ton risque prévu : ce que chaque point de risque t'a rapporté.",
        "Utile pour te comparer aux autres participants : quelqu'un qui fait +6 % avec trois fois ton risque n'est pas meilleur que toi à +3 %.",
        "+3,4 % de performance pour 10,3 % de risque : 0,33. Un concurrent à +5 % avec 25 % de risque : 0,20."),
    # ------------------------------------------------------------------ chance ou talent
    "percentile monte carlo": (
        "Ta place parmi des milliers de portefeuilles tirés au hasard dans le même univers, sur la même période.",
        "50 % = tu fais comme un tirage au sort. Il faut être au-dessus de 95 % pour que le hasard devienne une explication peu crédible. « À risque égal » compare seulement aux tirages aussi risqués que toi.",
        "« Mieux que 72 % » : sur 10 000 portefeuilles au hasard, 2 800 font mieux que toi. C'est honorable, mais un singe sur quatre te bat."),
    "jumeaux": (
        "Des portefeuilles « jumeaux » : même nombre de lignes, mêmes secteurs, mêmes poids, mais des titres tirés au sort dans chaque secteur.",
        "Te comparer à eux isole ta sélection de titres : tes choix de secteurs sont neutralisés. Au-dessus de 95 % : tes titres font la différence.",
        "Tu bats 60 % de tes jumeaux : à répartition sectorielle identique, tes titres font un peu mieux que le hasard, sans que ce soit prouvé."),
    "fausses decouvertes": (
        "Parmi tous les participants, combien battent vraiment l'indice, une fois retirés ceux qui n'ont eu que de la chance (Barras, Scaillet et Wermers, 2010).",
        "Sur 1 000 participants sans aucun talent, 50 passent quand même le test à 5 %. Il faut donc corriger avant de crier au talent.",
        "997 participants, 80 semblent battre l'indice au seuil de 5 % : environ 50 sont attendus par pur hasard. Seule une trentaine a peut-être quelque chose."),
    "recouvrement": (
        "La part de ton portefeuille qui est identique à l'indice (le complément de l'active share).",
        "Haut = tu ressembles à l'indice, tu ne peux pas beaucoup t'en écarter. Bas = tu fais vraiment autre chose.",
        "Recouvrement de 15 % : 85 % de ton portefeuille diffère de l'indice."),
    # ------------------------------------------------------------------ momentum et signaux
    "momentum 12-1": (
        "Le rendement du titre sur les 12 derniers mois en laissant de côté le dernier mois (Jegadeesh et Titman, 1993). Le dernier mois est exclu parce qu'il a tendance à s'inverser.",
        "Positif et élevé = tendance haussière, qui se prolonge en moyenne quelques mois. Négatif = tendance baissière : les perdants continuent souvent de perdre.",
        "Le titre valait 100 € il y a 12 mois, 130 € il y a un mois et 125 € aujourd'hui : momentum 12-1 = 130 ÷ 100 − 1 = +30 % (la baisse du dernier mois ne compte pas)."),
    "momentum residuel": (
        "Le momentum une fois retiré ce que le marché explique : le rendement propre du titre sur 12 mois, divisé par son incertitude (Blitz, Huij et Martens, 2011).",
        "Au-dessus de 2 : le titre a monté par lui-même, pas seulement porté par le marché. Proche de 0 : sa hausse vient de son bêta.",
        "Deux titres ont fait +40 %. Le premier a un bêta de 2 dans un marché à +20 % : alpha nul. Le second a un bêta de 0,5 : il a gagné 30 points par lui-même, son alpha est fort."),
    "asymetrie de beta": (
        "Le bêta les jours où le marché monte, moins le bêta les jours où il baisse.",
        "Positif = le titre accompagne plus les hausses que les baisses (le profil rêvé). Négatif = il encaisse plus les baisses. Signal très bruité : chaque bêta ne repose que sur la moitié des jours.",
        "Bêta de 1,2 les jours de hausse et 0,8 les jours de baisse : asymétrie = +0,4."),
    "plus haut 52 semaines": (
        "Le cours actuel divisé par son plus haut de l'année (George et Hwang, 2004).",
        "Proche de 100 % = le titre est à son sommet. Contrairement à l'intuition, ces titres continuent en moyenne de bien faire : les investisseurs hésitent à acheter « au plus haut ».",
        "Cours à 92 €, plus haut de l'année à 100 € : 92 %."),
    "moyenne 10 mois": (
        "Le cours divisé par sa moyenne des 10 derniers mois (règle de Faber, 2007).",
        "Au-dessus de 1 : tendance haussière. En dessous : tendance baissière, un signal de prudence simple et documenté.",
        "Cours à 48 €, moyenne sur 10 mois à 50 € : 0,96, le titre est sous sa tendance."),
    "regularite": (
        "La régularité de la hausse (« frog in the pan », Da, Gurun et Warachka, 2014) : le titre a-t-il monté par petites touches fréquentes ou par un seul gros saut ?",
        "Une hausse régulière se prolonge mieux qu'une hausse due à une seule annonce : les investisseurs la remarquent moins, donc elle est moins vite corrigée.",
        "Deux titres à +30 % sur un an. L'un a monté 150 jours sur 250 : régulier. L'autre a tout fait en une journée de résultats : momentum fragile."),
    "ic": (
        "Le coefficient d'information : la corrélation entre le classement des titres par le signal et leur rendement le mois suivant.",
        "0 = le signal ne prédit rien. 0,05 est déjà un bon signal en gestion quantitative. Ce qui compte : qu'il soit positif la plupart des mois.",
        "IC moyen de 0,04 et positif 60 % des mois : un avantage faible mais réel, qui ne se voit que sur beaucoup de titres et beaucoup de mois."),
    "rotation": (
        "La part du portefeuille renouvelée à chaque reconstruction mensuelle.",
        "Haut = beaucoup d'ordres, donc beaucoup de frais dans la vraie vie. Un modèle qui ne bat l'indice qu'avant frais ne vaut rien.",
        "Rotation de 23 % par mois : presque un quart du portefeuille change chaque mois, soit près de trois fois le portefeuille par an."),
    "confirmations": (
        "Le nombre de tests indépendants que le modèle réussit (face au hasard, par sous-périodes, avec des réglages voisins, après retrait des facteurs).",
        "Un test réussi ne prouve rien. C'est leur accord qui compte : il en faut presque tous pour prendre le modèle au sérieux.",
        "« 2 sur 7 » : le modèle a eu de la chance sur une période ou un réglage. À ne pas suivre."),
    "reglages essayes": (
        "Le nombre de versions du modèle que tu as testées. Plus on essaie, plus on trouve par hasard une version qui semble marcher.",
        "Ce nombre relève la barre que le Sharpe doit franchir. Il rend le test honnête : il t'empêche de garder le meilleur de vingt essais comme s'il était le seul.",
        "Après 20 essais, le meilleur d'entre eux affiche un Sharpe flatteur même si aucun n'a de valeur."),
    # ------------------------------------------------------------------ valorisation et résultats
    "per": (
        "Le cours divisé par le bénéfice par action : le nombre d'années de bénéfices que tu paies.",
        "Bas = bon marché, ou bénéfices menacés. Haut = le marché attend de la croissance. À comparer au secteur et à la croissance. Trompeur pour les cycliques : il est bas au sommet du cycle.",
        "Action à 60 €, bénéfice de 4 € par action : PER = 15. Tu paies 15 ans de bénéfices actuels."),
    "peg": (
        "Le PER divisé par la croissance attendue des bénéfices, en % (Lynch).",
        "Sous 1 : la croissance est bon marché. Entre 1 et 2 : prix correct. Au-dessus de 2 : tu paies cher la croissance.",
        "PER de 20 et croissance attendue de 10 % par an : PEG = 2. Le même PER avec 25 % de croissance : PEG = 0,8."),
    "lynch": (
        "Le ratio de Lynch : (croissance des bénéfices + rendement du dividende) divisé par le PER. Il sert pour les sociétés à croissance lente qui versent un dividende.",
        "Au-dessus de 1 : correct. Au-dessus de 2 : attractif. Sous 1 : cher pour ce que ça rapporte.",
        "Croissance de 6 %, dividende de 4 %, PER de 8 : (6 + 4) ÷ 8 = 1,25."),
    "roe": (
        "La rentabilité des capitaux propres : le bénéfice divisé par l'argent des actionnaires.",
        "Au-dessus de 10 % : l'entreprise crée de la valeur. Au-dessus de 20 % : très rentable. À vérifier avec la dette : beaucoup de dette gonfle le ROE.",
        "Bénéfice de 150 millions pour 1 milliard de capitaux propres : ROE = 15 %."),
    "dette nette / ebitda": (
        "La dette (moins la trésorerie) divisée par le résultat d'exploitation avant amortissements : le nombre d'années pour rembourser.",
        "Sous 1,5 : bilan solide. Au-dessus de 3 : fragile si les taux montent ou si l'activité ralentit. Sans objet pour les banques et les assureurs.",
        "2 milliards de dette nette, 800 millions d'Ebitda : 2,5 années."),
    "rentabilite brute": (
        "La marge brute divisée par le total de l'actif (Novy-Marx, 2013) : ce que l'entreprise gagne avec les moyens qu'elle mobilise.",
        "Plus c'est haut, mieux c'est. Les entreprises très rentables font mieux en bourse sur la durée, même quand elles paraissent chères.",
        "400 millions de marge brute pour 1 milliard d'actif : 40 %, une entreprise très rentable."),
    "croissance de l'actif": (
        "La variation du total du bilan sur un an (Cooper, Gulen et Schill, 2008).",
        "Une forte hausse est un mauvais signe en moyenne : les entreprises qui grossissent vite (rachats, gros investissements) déçoivent ensuite. Modéré = sain.",
        "Actif passé de 10 à 13 milliards après un gros rachat : +30 %, à surveiller."),
    "emissions": (
        "La variation du nombre d'actions sur un an (Pontiff et Woodgate, 2008).",
        "Négatif = l'entreprise rachète ses actions : bon signe. Positif = elle en émet et dilue les actionnaires : mauvais signe en moyenne.",
        "−2 % : chaque action représente une part un peu plus grande de l'entreprise. +8 % : ta part a fondu de 8 %."),
    "surprise": (
        "L'écart entre le bénéfice publié et ce que les analystes attendaient, en %.",
        "Positive = bonne surprise. Le cours continue en moyenne dans le même sens pendant deux à trois mois : c'est la dérive post-annonce (Bernard et Thomas, 1989).",
        "Attendu 2,00 € par action, publié 2,20 € : surprise de +10 %. Le titre a tendance à faire mieux que le marché dans les semaines qui suivent."),
    "revision": (
        "De combien les analystes ont changé leur prévision de bénéfice par action depuis 90 jours.",
        "Positive = les analystes relèvent leurs chiffres : le signal fondamental le plus fiable à court terme (Chan, Jegadeesh et Lakonishok, 1996). Sous −5 % : alerte.",
        "Prévision passée de 5,00 € à 5,30 € : +6 %. Passée de 5,00 € à 4,60 € : −8 %, le consensus décroche."),
    "rendement du fcf": (
        "Le flux de trésorerie disponible (ce qui reste en caisse après les investissements) divisé par la valeur en bourse.",
        "Au-dessus de 5 % : l'entreprise produit beaucoup de cash pour son prix. Négatif : elle en consomme.",
        "500 millions de trésorerie disponible, 10 milliards de valeur en bourse : 5 %."),
    "croisement": (
        "Le verdict qui croise le prix (momentum) et les bénéfices (révisions des analystes).",
        "« Tendance confirmée » : le cours monte et les bénéfices suivent. « Hausse sans soutien » : le cours monte seul, fragile. « À étudier » : bénéfices en hausse, cours en retard. « À éviter » : les deux baissent.",
        "Momentum de +40 % mais révisions à −6 % : « Hausse sans soutien ». Le marché achète une histoire que les chiffres ne confirment pas encore."),
    # ------------------------------------------------------------------ portefeuille et ordres
    "latent": (
        "Le gain ou la perte sur une ligne que tu n'as pas encore vendue : (cours actuel − prix de revient) × quantité.",
        "Tant que tu n'as pas vendu, rien n'est acquis. Attention au biais : une ligne en perte n'a aucune raison de « revenir à ton prix d'achat ».",
        "58 actions achetées 36,24 €, cours à 38,00 € : latent = (38,00 − 36,24) × 58 = +102 €."),
    "prix de revient": (
        "Le prix moyen auquel tu as acheté la ligne, tous achats confondus.",
        "Il sert à calculer ton gain, pas à décider : le marché ne connaît pas ton prix d'achat.",
        "10 actions à 50 € puis 10 à 60 € : prix de revient = 55 €."),
    "poids": (
        "La part de la ligne dans la valeur totale du portefeuille.",
        "Tes règles : 10 % au maximum par ligne, 20 % par secteur, 30 % par place de cotation. Un poids qui dérive loin de sa cible appelle un rééquilibrage.",
        "2 100 € de Klépierre dans un portefeuille de 51 800 € : 4,1 %."),
    "ecart": (
        "La différence entre le poids actuel et le poids cible de la ligne.",
        "Positif = la ligne a trop grossi, à alléger. Négatif = à renforcer. On ne rééquilibre que les écarts qui valent les frais.",
        "Cible à 4,0 %, poids actuel à 5,2 % : écart de +1,2 point, soit environ 620 € à vendre sur 51 800 €."),
    "cash": (
        "L'argent non investi.",
        "Il ne rapporte rien dans le concours mais amortit les baisses et permet de saisir une occasion.",
        "8 € de cash sur 51 800 € : tu es investi à 100 %. Pour acheter, il faudra d'abord vendre."),
    # ------------------------------------------------------------------ scénarios politiques
    "sensibilite facteur france": (
        "Ce que la ligne (ou le portefeuille) perd quand le CAC 40 fait 1 point de moins que l'Europe, une fois son bêta de marché retiré.",
        "0 = insensible au risque politique français. 1 = réagit comme le CAC 40. Seules les sensibilités avec un t d'au moins 2 comptent.",
        "Sensibilité de 0,3 : si la France décroche de 5 points face à l'Europe, ton portefeuille perd 0,3 × 5 = 1,5 point de plus que prévu."),
    "ecart implicite": (
        "Ce que le marché semble attendre comme différence totale sur le facteur France entre les deux issues, déduit des jours où la probabilité a bougé.",
        "Négatif = le marché craint le scénario de tension. À ne lire que si la statistique t dépasse 2,6.",
        "Quand la probabilité monte de 10 points, le CAC 40 perd 0,8 % face à l'Europe : écart implicite = −8 % entre « exclu » et « certain »."),
    "evenement binaire": (
        "Le gain ou la perte attendu si la probabilité change, sachant qu'une partie du choc est déjà dans les cours.",
        "Les cours ne bougent que sur la surprise. Pondéré par les probabilités, le total est nul : c'est une mesure d'exposition, pas une prévision.",
        "Choc total de −10 %, probabilité de 30 % : 3 % sont déjà payés. Si le scénario se réalise : −7 %. S'il est écarté : +3 %. Et 0,3 × (−7) + 0,7 × 3 = 0."),
    # ------------------------------------------------------------------ divers
    "rendement annualise": (
        "Le rendement ramené à un rythme annuel, pour comparer des périodes de longueurs différentes.",
        "Sur moins d'un an, il extrapole : +3 % en un mois donne +43 % « annualisé », ce qui ne veut rien dire. À ne prendre au sérieux que sur plusieurs années.",
        "+33 % en trois ans : (1,33)^(1/3) − 1 = +10 % par an."),
    "performance depuis le depart": (
        "Le gain total depuis le début du concours, en % du capital apporté (50 160,50 €).",
        "C'est le chiffre du classement brut. À comparer à l'Euronext 100 sur la même période : faire +3 % quand l'indice fait +5 % n'est pas une réussite.",
        "Portefeuille à 51 845 € pour 50 160,50 € apportés : +1 684 €, soit +3,36 %."),
    "gain du jour": (
        "Ce que le portefeuille a gagné ou perdu depuis la clôture de la veille, en euros.",
        "À comparer à l'indice affiché dessous : avec un bêta de 0,67, tu dois gagner moins que lui les jours de hausse et perdre moins les jours de baisse.",
        "Indice à +0,75 %, toi à +0,43 % : c'est conforme à ton bêta, pas une contre-performance."),
    "secteurs couverts": (
        "Le nombre de secteurs (sur les 11 de la classification standard) où tu as au moins une ligne.",
        "Plus tu en couvres, moins tu dépends d'un seul pan de l'économie. « Plus gros » : le poids de ton premier secteur, à garder sous 20 % selon tes règles.",
        "11 sur 11, plus gros à 24 % : tu es partout, mais un secteur dépasse ta limite."),
    "univers a poids egaux": (
        "Le rendement qu'on obtient en achetant tous les titres de l'univers à parts égales, sans rien choisir.",
        "C'est la référence honnête d'une sélection : si tes choix ne battent pas « tout acheter », ils n'ont rien apporté (DeMiguel, Garlappi et Uppal, 2009).",
        "Modèle à +13,3 % par an, poids égaux à +14,7 % : le modèle a fait moins bien que l'absence de choix."),
    "score": (
        "La note composite du titre : la moyenne de ses signaux, chacun exprimé en écarts à la moyenne de l'univers et pondéré par tes curseurs.",
        "0 = dans la moyenne. +1 = nettement au-dessus. Seul le classement compte, pas la valeur exacte.",
        "Momentum à +1,5 écart-type et faible volatilité à −0,3, à poids égaux : score = (1,5 − 0,3) ÷ 2 = +0,6."),
    "volume": (
        "Le montant échangé en moyenne chaque jour sur le titre.",
        "Plus il est élevé, plus tu peux acheter ou vendre sans faire bouger le cours. Sous 1 million d'euros par jour : titre peu liquide.",
        "5 millions d'euros par jour : un ordre de 2 000 € passe inaperçu (0,04 % du volume)."),
    "tirages qui battent l'indice": (
        "La part des portefeuilles tirés au hasard qui font mieux que l'indice sur la période.",
        "Si elle est haute, battre l'indice ne prouve rien : c'était facile sur cette période (souvent parce que les petites valeurs ont bien marché).",
        "62 % des tirages battent l'indice : toi aussi, mais comme près de deux singes sur trois."),
    "poids couvert": (
        "La part de ton portefeuille actuel dont les titres étaient déjà cotés pendant l'épisode rejoué.",
        "Sous 80 %, le chiffre de l'épisode est à prendre avec prudence : il manque des lignes.",
        "Poids couvert de 96 % : une seule petite ligne n'existait pas à l'époque."),
    "facteur france": (
        "L'écart de rendement entre le CAC 40 et l'indice européen : ce que la France fait de plus ou de moins que l'Europe.",
        "Négatif pendant un épisode = les investisseurs ont fui la France en particulier, signe d'un risque politique ou budgétaire propre au pays.",
        "CAC 40 à −6 % et Europe à −2 % sur la même semaine : facteur France ≈ −4 %."),
    "probabilite du scenario de tension": (
        "La somme des probabilités que le marché de prédiction donne aux issues que tu as cochées comme inquiétantes pour les marchés.",
        "C'est un prix, pas un sondage : il réagit vite mais peut être déplacé par quelques gros parieurs, surtout loin du vote.",
        "Tu coches deux candidats cotés 22 % et 12 % : probabilité du scénario = 34 %."),
    "croissance": (
        "La croissance attendue des bénéfices par action, en % par an.",
        "C'est une prévision d'analystes, donc incertaine. Elle sert à juger si le PER est justifié (voir le PEG).",
        "Bénéfice par action attendu de 5,50 € l'an prochain contre 5,00 € cette année : +10 %."),
    "signaux": (
        "Le décompte des signaux favorables et défavorables du titre sur la page (qualité, révisions, surprise, émissions…).",
        "« 3 + / 1 − » : trois signaux au vert, un au rouge. Ce n'est pas une note : un seul signal rouge grave peut suffire à écarter un titre.",
        "Rentabilité élevée, révisions en hausse, rachats d'actions, mais dernière surprise négative : « 3 + / 1 − »."),
    "ecarts aux regles": (
        "Le nombre de tes règles d'entrée que la ligne ne respecte pas.",
        "0 = conforme. Une ligne déjà détenue qui perd une règle est à réexaminer au prochain rééquilibrage, pas à vendre dans la minute.",
        "Momentum devenu négatif et dette au-dessus de 3 : 2 écarts."),
    "perte estimee": (
        "Ce que ton portefeuille actuel aurait perdu si la crise rejouée se reproduisait à l'identique.",
        "C'est un ordre de grandeur : la prochaine crise ne ressemblera pas exactement à la précédente. À comparer à la perte de l'indice sur le même épisode.",
        "Crise de 2020 rejouée : −24 % pour toi, −35 % pour l'indice. Ton bêta bas t'aurait fait perdre 11 points de moins."),
    # ------------------------------------------------------------------ portefeuille modèle : scénarios
    "stabilite de la selection": (
        "La fréquence moyenne à laquelle les titres retenus le restent quand on remélange 200 fois les mois de l'année écoulée et qu'on refait la sélection (rééchantillonnage, Michaud, 1998).",
        "Au-dessus de 75 % : la sélection ne dépend pas d'un accident du calendrier. Sous 60 % : avec une année à peine différente, le modèle aurait choisi un autre portefeuille.",
        "Stabilité de 74 % : en moyenne, un titre retenu aujourd'hui l'aurait été dans 148 scénarios sur 200."),
    "frequence de selection": (
        "La part des 200 scénarios dans lesquels le titre est retenu par le modèle.",
        "80 % ou plus : choix solide. Moins de 50 % : le titre doit sa place à quelques semaines de cours. Un « remplaçant » à 70 % vaut mieux qu'un titulaire à 40 %.",
        "Un titre retenu dans 180 scénarios sur 200 : 90 %, il serait choisi presque quoi qu'il arrive. Un autre dans 60 sur 200 : 30 %, un accident des données."),
    "poids deplaces": (
        "La part du portefeuille qui change quand on remplace les poids calculés sur la vraie année par la moyenne des poids calculés sur les 200 scénarios.",
        "Sous 10 % : les poids sont stables. Au-dessus de 20 % : l'optimiseur réagit fortement à de petites différences de données, ses poids sont à prendre avec prudence.",
        "4 % : en passant aux poids moyens des scénarios, seuls 4 % du portefeuille changeraient de ligne."),
    "mauvais cas": (
        "Le 5e centile des trajectoires simulées : le résultat qui n'est pire que dans une trajectoire sur vingt.",
        "C'est le chiffre le plus utile de la simulation : il dit ce que tu risques si les prochains mois tournent mal, sans être le scénario catastrophe.",
        "Mauvais cas à −9 % sur 50 000 € : une fois sur vingt, tu finirais le concours avec 4 500 € de moins qu'aujourd'hui, ou pire."),
    "cas median": (
        "Le résultat du milieu : la moitié des trajectoires simulées finissent au-dessus, l'autre moitié en dessous.",
        "Ce n'est pas une prévision. Avec l'hypothèse « aucun avantage », il reflète seulement le rendement moyen passé de l'indice.",
        "Cas médian à +6 % : sur 5 000 trajectoires, 2 500 font mieux que +6 % et 2 500 font moins bien."),
    "bon cas": (
        "Le 95e centile des trajectoires simulées : le résultat qui n'est dépassé que dans une trajectoire sur vingt.",
        "L'écart entre le bon et le mauvais cas mesure l'incertitude. Plus la fourchette est large, moins le résultat final dépend de toi.",
        "Mauvais cas −9 %, bon cas +25 % : sur les mois qui restent, la chance peut déplacer ton résultat de 34 points."),
    "risque de finir en perte": (
        "La part des trajectoires simulées qui finissent sous la valeur d'aujourd'hui.",
        "Même un bon portefeuille d'actions a souvent 25 à 40 % de risque de perdre sur quelques mois. C'est normal, pas un défaut.",
        "26 % : dans une trajectoire sur quatre, tu termines le concours plus bas qu'aujourd'hui."),
    "baisse en cours de route": (
        "La plus forte baisse depuis un sommet subie pendant la trajectoire (ici : celle de la trajectoire médiane).",
        "C'est ce qu'il faudra supporter sans paniquer, même si la fin est bonne. À comparer entre portefeuilles : le plus calme est celui qu'on garde le plus facilement.",
        "−7,6 % : dans une trajectoire typique, tu verras à un moment ton portefeuille 7,6 % sous son plus haut."),
    "bat l'indice": (
        "La part des trajectoires simulées où le portefeuille finit devant l'Euronext 100.",
        "50 % = pile ou face. Avec l'hypothèse « aucun avantage », un chiffre un peu au-dessus de 50 % vient seulement d'une volatilité plus basse, pas d'un talent.",
        "56 % : sur 5 000 trajectoires, 2 800 finissent devant l'indice."),
}

# Libellé du site (normalisé) -> entrée. Les libellés identiques à une clé n'ont pas besoin d'y figurer.
ALIAS = {
    "vol. prevue": "volatilite prevue", "vol ewma": "volatilite prevue",
    "volatilite realisee": "volatilite", "vol. du marche": "volatilite", "vol. du momentum": "volatilite", "volatilite du facteur momentum": "volatilite",
    "diversification": "ratio de diversification", "diversif.": "ratio de diversification",
    "nombre effectif de paris": "paris independants", "paris reellement independants": "paris independants",
    "var 95 % sur un jour": "var 95", "var 95 % cornish-fisher": "var 95", "var 95 %": "var 95",
    "perte moyenne au-dela": "perte moyenne au-dela",
    "baisse max": "plus forte baisse", "plus forte baisse depuis un sommet": "plus forte baisse", "recul depuis le plus haut": "plus forte baisse",
    "recul du facteur depuis son plus haut": "plus forte baisse",
    "δ volatilite": "delta volatilite", "δ diversification": "delta diversification",
    "corr. avec ton portefeuille": "correlation", "corr. max avec une ligne": "correlation", "correlation < 0,7": "correlation",
    "part du volume d'un jour": "jours pour tout vendre",
    "turbulence actuelle": "turbulence", "seuil turbulent": "turbulence",
    "part du premier facteur": "ratio d'absorption",
    "poids a donner au momentum": "exposition conseillee",
    "sharpe": "ratio de sharpe", "sharpe annualise": "ratio de sharpe",
    "probabilite d'un vrai sharpe positif": "probabilite que le vrai sharpe soit positif",
    "alpha annualise": "alpha", "alpha une fois les facteurs retires": "alpha",
    "t de l'alpha": "statistique t", "t": "statistique t",
    "p-valeur bootstrap": "p-valeur", "p-valeur classique": "p-valeur", "meme calcul en supposant la loi normale": "p-valeur",
    "part expliquee par les facteurs": "r2", "r² ajuste face au marche": "r2", "part des mouvements expliquee": "r2",
    "effet selection": "effet selection", "allocation": "effet allocation", "selection": "effet selection",
    "asymetrie des rendements": "asymetrie",
    "jarque-bera, p": "test de normalite", "shapiro-wilk, p": "test de normalite",
    "pgr": "pgr", "plr": "pgr", "pgr − plr": "pgr",
    "gain sur une prevision naive": "score de brier",
    "performance ÷ risque prevu": "performance sur risque",
    "mieux que": "percentile monte carlo", "a risque egal, mieux que": "percentile monte carlo", "ta place parmi les tirages": "percentile monte carlo",
    "mieux que tes jumeaux dans": "jumeaux",
    "vraiment meilleurs que l'indice": "fausses decouvertes", "vraiment moins bons": "fausses decouvertes", "sans talent": "fausses decouvertes",
    "recouvrement avec le marche": "recouvrement",
    "momentum 12-7": "momentum 12-1", "score momentum": "momentum 12-1",
    "alpha": "alpha", "asym. beta": "asymetrie de beta",
    "% du plus haut 52 s.": "plus haut 52 semaines", "cours / moyenne 10 mois": "moyenne 10 mois",
    "ic moyen": "ic", "mois ou ic > 0": "ic",
    "rotation mensuelle": "rotation",
    "roe ≥ 10 %": "roe", "dette ≤ 3": "dette nette / ebitda",
    "rentab.": "rentabilite brute", "rentabilite brute mediane": "rentabilite brute",
    "actif 1 an": "croissance de l'actif",
    "derniere surprise": "surprise", "derive post-annonce favorable": "surprise", "derive": "surprise",
    "revision bpa 90 j": "revision", "bpa 90 j": "revision",
    "latent %": "latent", "latent €": "latent",
    "poids avant": "poids", "poids apres": "poids", "ton poids": "poids",
    "sensibilite": "sensibilite facteur france", "sensibilite du portefeuille au facteur france": "sensibilite facteur france",
    "ecart implicite entre les deux issues": "ecart implicite",
    "si la probabilite monte de 10 points": "evenement binaire", "si le scenario de tension se realise": "evenement binaire", "s'il est ecarte": "evenement binaire",
    "cash apres": "cash",
    "titres solides": "frequence de selection", "titres fragiles": "frequence de selection", "poids deplaces par les scenarios": "poids deplaces",
    "le modele finit devant toi dans": "bat l'indice", "ecart dans le mauvais cas": "mauvais cas",
    "depuis le depart": "performance depuis le depart", "performance brute": "performance depuis le depart",
    "volume moyen": "volume", "montant echange par jour": "volume",
    "ecarts": "ecarts aux regles", "perte estimee du portefeuille": "perte estimee", "impact estime sur le portefeuille": "perte estimee",
    "ton portefeuille actuel": "perte estimee", "bpa 1 an": "croissance", "chiffre d'affaires 1 an": "croissance",
    "bat les poids egaux": "univers a poids egaux", "ecart avec les poids egaux": "univers a poids egaux", "rendement univers": "univers a poids egaux",
}
# Cas particulier : « Alpha (t) » est le momentum résiduel (la parenthèse disparaît à la normalisation).
SPECIAUX = {"Alpha (t)": "momentum residuel", "Momentum résiduel": "momentum residuel"}


def _norm(libelle):
    s = unicodedata.normalize("NFD", str(libelle))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"\([^)]*\)", "", s)
    return re.sub(r"\s+", " ", s).strip(" :")


def bulle(libelle):
    """Le « i » en HTML pour un libellé, ou une chaîne vide. Le panneau s'ouvre au survol (ou au toucher) en bas de l'écran."""
    import html as _h
    a = aide(libelle)
    if not a:
        return ""
    quoi, lire, ex = (_h.escape(x) for x in a)
    return ('<span class="nw-i" tabindex="0">i<span class="nw-tip"><span class="h">' + _h.escape(str(libelle)) + '</span>'
            '<span class="l">C\'est quoi</span>' + quoi + '<span class="l">Comment le lire</span>' + lire +
            '<span class="l">Exemple</span>' + ex + '</span></span>')


def aide(libelle):
    """Renvoie (c'est quoi, comment le lire, exemple) pour un libellé du site, ou None s'il n'a pas besoin d'explication."""
    if libelle in SPECIAUX:
        return G[SPECIAUX[libelle]]
    n = _norm(libelle)
    cle = n if n in G else ALIAS.get(n)
    return G.get(cle) if cle else None
