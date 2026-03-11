# 🏀 Analyse statistique du momentum dans le sport — NBA

---

## Questions directrices

> *Comment les runs faits par une équipe influencent-ils la victoire ?*
> *À quelle fréquence peut-on observer des runs ?*
> *Comment les cotes comparatives peuvent-elles être reliées à la victoire ?*

**Variables d'intérêt :**
- Différentiel de score
- Points marqués par match
- Efficacité au tir
- Cotes des paris NBA

---

## 📌 Définition d'un Run

### Niveau 1 — Run de points bruts

Séquence de points marqués **consécutivement par une seule équipe**.

| Moment | Score | Interprétation |
|--------|-------|----------------|
| t₁ | 10 – 0 | Run de 10 pts |
| t₂ | 15 – 2 | Run qui ralentit |
| t₃ | 20 – 6 | Run de 14 pts au total |

**Critère proposé :** écart relatif > 10 pts en moins de 3 minutes *(à affiner selon les données)*

---

### Niveau 2 — Score pondéré par actions positives

| Action | Poids |
|--------|-------|
| Panier à 2 pts | +2.0 |
| Panier à 3 pts | +3.0 |
| Lancer franc réussi | +1.0 |
| Rebond | +0.5 |
| Interception | +1.0 |
| Contre | +0.75 |

Un run correspond alors à une **accumulation rapide** de ce score pondéré. *(Modèle en cours d'affinage)*

---

## Partie 1 — Modèles ARMA / Séries temporelles

**Objectif :** Modéliser le différentiel de score match par match comme une série temporelle.

- Le différentiel de score par match est-il **stationnaire** ?
- Les performances passées d'une équipe **prédisent-elles** ses performances futures ? *(coefficients AR significatifs)*
- Les chocs (défaite surprise, blessure) **persistent-ils** ? *(coefficients MA significatifs)*
- Comparer les ordres `(p, q)` entre équipes pour identifier celles où le **momentum est le plus persistant**

> **Interprétation attendue :** si φ₁ > 0 et significatif → momentum positif (une bonne perf. en entraîne une autre). Si la série est un bruit blanc → pas de momentum modélisable.

---

## Partie 2 — Copules

### Q1 — Dépendance offensive inter-équipes
Quand une équipe marque beaucoup dans un match, l'adversaire marque-t-il aussi **plus que d'habitude** ?

### Q2 — Back-to-back
Les équipes NBA jouent ~82 matchs en 6 mois, avec environ **15 back-to-backs** par saison (deux matchs consécutifs, J et J+1).
Quelle est la structure de dépendance sur les résultats ?

### Q3 — Impact des prolongations (overtime)
Les matchs en OT entraînent-ils de **moins bonnes performances** dans les matchs suivants ?

---

## Partie 3 — Théorie des valeurs extrêmes

### Q1 — Probabilité d'observer un run extrême
Quelle est la probabilité d'un run exceptionnel (niveau brut ou score pondéré) ?

### Q2 — Distribution des plus grands écarts de score (blowouts)
- Quelle loi ajuster sur les **maxima de différentiel** de score ?
- Une équipe a-t-elle tendance à **répéter** ce type de performances sur une saison ?
- Cas symétrique : séries de **défaites sévères** (mauvaises équipes)

### Q3 — Overtime et extrêmes
Fréquence des prolongations, distribution des scores en OT, impact sur les matchs suivants.

---

## Partie 4 — Processus de Hawkes

**Hypothèse centrale :** le scoring en NBA est un processus **auto-excité** — chaque événement augmente temporairement la probabilité du suivant.

### Q1 — Auto-excitation du scoring
Un panier augmente-t-il temporairement la probabilité d'un **nouveau panier** (ou d'une action positive) ?

### Q2 — Durée du momentum
Quelle est la **durée moyenne** d'un effet de momentum ?
*(Pour aller plus loin : varie-t-elle selon le style de jeu de l'équipe ?)*

### Q3 — Influence mutuelle des deux équipes
Les événements de scoring de l'équipe A **inhibent-ils ou excitent-ils** le scoring de l'équipe B ?
→ Processus de Hawkes **bivarié**

---

## Partie 5 — Style de jeu et runs

**Définition du style de jeu :** classement des équipes selon leurs ratios relatifs par rapport à la moyenne de la ligue.

| Style | Indicateurs |
|-------|-------------|
| 🛡️ Défensif | Peu de points encaissés |
| ⚔️ Agressif | Beaucoup de rebonds + interceptions |
| 🎯 Offensif | Beaucoup de points marqués |
| 🌐 Moderne | Beaucoup de tentatives à 3 pts |

**Question centrale :**
- Quel style de jeu **génère le plus de runs** ?
- Le momentum est-il une propriété du **style** ou de **l'équipe** ?
