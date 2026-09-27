# Challenge Taux de conversion Abonnement Data Science Weekly

## Certification CDSD — Bloc 3 — RNCP35288
**Analyse prédictive de données structurées par l'intelligence artificielle**

Patrick Grouillet · Jedha Fullstack Data Science · 1er octobre 2026

---

## Contexte

Les data scientists de [datascienceweekly.org](https://www.datascienceweekly.org/) souhaitent prédire si un visiteur de leur site va s'abonner a la newsletter, à partir de quelques informations sur l'utilisateur. Ce projet s'inscrit dans un challenge de machine learning ou les équipes soumettent leurs prédictions et sont classées selon le **F1-score**.

## Dataset

| Fichier | Description |
|---------|-------------|
| `conversion_data_train.csv` | 284 580 lignes, 6 variables + cible `converted` |
| `conversion_data_test.csv` | 31 620 lignes, 6 variables (sans cible) |

**Variables** : 'country', 'age', 'new_user', 'source', 'total_pages_visited', 'converted'

**Déséquilibre** : seulement 3.23% de conversions (ratio 30:1)

## Approche

1. **EDA approfondie** : analyse de la cible, variables catégorielles, numériques, corrélations, outliers
2. **Feature engineering** : création de 4 features ('jeunes', 'chinois', 'bcppages', 'réengagement')
3. **Préprocessing** : 'ColumnTransformer' avec 'StandardScaler', 'OneHotEncoder (drop='first')', passthrough pour les binaires
4. **Modèles comparés** : Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, AdaBoost
5. **Optimisation** : `GridSearchCV` avec `StratifiedKFold` (3 splits) sur Gradient Boosting et Random Forest
6. **Prédictions finales** : réentrainement du meilleur modèle sur toutes les données, prédictions sur le test set

## Résultats 

| Modèle | F1 Test | AUC |
|--------|---------|-----|
| **Gradient Boosting** | **0.7549** | **0.9854** |
| AdaBoost | 0.7408 | — |
| Random Forest | 0.5795 | — |
| Decision Tree | 0.5049 | — |
| Logistic Regression | 0.5113 | — |

**Best hyperparamètres** : 'learning_rate=0.1', 'max_depth=5', 'n_estimators=200', 'subsample=0.8'

**Features les plus importantes** : 'total_pages_visited' (76.2%), 'réengagement' (9.1%), 'chinois' (5.8%), 'age' (5.4%)

## Recommandations business

1. **Engagement** : améliorer la navigation du site, placer les bulletins d'inscription après 8-10 pages
2. **Fidélisation** : mécanismes de ré-engagement pour fidéliser les anciens abonnés
3. **Ciblage jeune** : orienter le marketing vers les 17-25 ans
4. **Marché chinois** : rechercher les causes du taux quasi-nul du marché chinois (0.13%)
5. **Source de trafic** : optimiser les campagnes publicitaires

## Outils principaux

- Python 3, Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
