# Entraînement en moins d'une heure : protocole GPU

Le programme `src/train_fast.py` vise un budget total de **45 minutes**, avec une réserve pour l'évaluation. Il s'arrête aussi si les 12 époques prévues sont terminées avant. Ce budget est un objectif de temps mural, pas une garantie en cas de blocage du système ou du réseau.

## Conformité aux consignes

- Même dataset Drug Review UCI autorisé par le professeur.
- Baseline Vanilla et deux cellules distinctes de l'annexe : CIFG et WMC.
- Cellules toujours écrites à la main avec `nn.Linear` ; aucun `nn.LSTM`.
- Mêmes partitions train/validation/test et même protocole pour les trois architectures.
- Sous-échantillonnage stratifié : **15 000 train, 1 500 validation, 3 000 test** après déduplication exacte. Le professeur permet de réduire les données si nécessaire.
- Nombre de paramètres, accuracy, F1 macro, F1 par classe et matrices de confusion enregistrés.
- Sélection de l'époque et de la configuration au F1 macro de validation. Le test est évalué après sélection ; les résultats ne sont pas utilisés pour ajuster les hyperparamètres.

## Ce qui change par rapport à la première étude

Les résultats préliminaires ont été retirés pour conserver uniquement l'expérience finale.

Le nouveau classifieur moyenne les états cachés sur les tokens valides (`mean pooling`). Les cellules récurrentes restent exactement celles de l'annexe. Cette moyenne fournit au classifieur l'information de tout le texte plutôt que du dernier état uniquement. Elle est identique pour les trois modèles.

Autres choix communs : embeddings 64, état caché 64, au plus 160 tokens, dropout 0,2, initialisation Xavier des couches linéaires et biais d'oubli à 1. Les embeddings sont appris depuis zéro sur le train. Aucun modèle de langage préentraîné n'est utilisé.

La transformation linéaire de l'entrée est calculée pour tous les tokens à la fois. Les tests vérifient que cela conserve les sorties et gradients de la version pas à pas.

## Les 12 expériences

Chaque cellule est comparée avec quatre réglages. L'optimiseur est Adam pour tous :

| Réglage | LR | Batch | Pondération de la perte |
|---|---:|---:|---|
| reference | 0,003 | 256 | aucune |
| learning_rate | 0,001 | 256 | aucune |
| batch_size | 0,003 | 128 | aucune |
| class_weight | 0,003 | 256 | racine de la fréquence inverse du train |

Chaque alternative modifie un seul paramètre de la référence. Le professeur demande de comparer les hyperparamètres, sans imposer ceux à comparer. La première étude comparait notamment Adam et RMSprop ; celle-ci étudie LR, batch et pondération des classes.

Chaque tour entraîne les **12 réglages pour une époque**. Un tour incomplet à l'échéance est exclu : les checkpoints retenus et les historiques correspondent toujours au même nombre d'époques complètes pour tous. Des checkpoints de différentes meilleures époques peuvent être retenus, selon le même critère de validation. Le changement de batch modifie le nombre d'updates : ce n'est pas une comparaison à nombre d'updates identique.

## Lancer sur ton ordinateur

Depuis `C:\Users\abder\Desktop\WORK\tp\nlp\projet_rnn`, après avoir installé la mise à jour du code :

```powershell
..\.venv\python.exe -m src.train_fast --device cuda --minutes 45 --output results/fast
```

Le programme affiche les scores de validation après chaque expérience et la durée de chaque tour. Il refuse de réutiliser un dossier contenant déjà `runs.json` pour ne pas remplacer les résultats. Choisis un autre `--output` pour une nouvelle exécution.

Pour laisser le programme comparer CPU et GPU sur des lots synthétiques avant de choisir le plus rapide :

```powershell
..\.venv\python.exe -m src.train_fast --device auto --minutes 45 --output results/fast_auto
```

Un GPU n'est pas automatiquement plus rapide pour de petites cellules écrites à la main. Un benchmark sur le laptop RTX 5060 a montré un avantage GPU pour le WMC de cette taille. Le programme n'utilise pas les labels de test pour ce benchmark.

## Après entraînement

- `comparison.csv` : comparaison finale des trois modèles sélectionnés.
- `runs.json` : 12 expériences, historique par époque et nombre de paramètres.
- `results.json` : scores test, référence majoritaire et matrices de confusion.
- `budget.json` : durée totale, nombre d'époques complètes et motif d'arrêt.
- `environment.json` : paramètres et matériel utilisé.
- `vocab.json` et `checkpoints/*.pt` : fichiers nécessaires à la prédiction.

Le programme de prédiction retrouve maintenant le vocabulaire à partir du checkpoint et vérifie son empreinte pour les nouveaux modèles. Les checkpoints de la première étude restent compatibles.

**70 % d'accuracy est un objectif, pas une promesse.** La référence majoritaire est proche de 66 %. Il faut également examiner le F1 macro et les erreurs sur les avis négatifs/neutres. Le GPU accélère le calcul ; il n'améliore pas les scores à protocole identique par lui-même.

Le rapport final `output/pdf/report.pdf` décrit cette étude GPU. Son texte est dans `output/pdf/report.md`. Le PowerPoint est laissé pour plus tard. Renseigner `report_details.json` pour ajouter les noms, le groupe et l'URL GitHub au rapport.

## Résultats mesurés sur le laptop RTX 5060

L'étude enregistrée dans `results/gpu_45min` a terminé **8 époques complètes pour chacun des 12 réglages**, en **37 minutes 49 secondes**, préparation et évaluation comprises dans la durée enregistrée par le programme. Le budget visé était de 45 minutes ; un nouveau tour n'a pas été lancé pour conserver la réserve de temps.

Mesures sur les mêmes **3 000 avis de test** :

| Cellule | Accuracy test | F1 macro test | Meilleure époque de validation |
|---|---:|---:|---:|
| Vanilla | 73,97 % | 0,5727 | 7 |
| CIFG | 75,63 % | 0,5662 | 4 |
| WMC | 74,60 % | 0,5505 | 4 |
| Référence majoritaire | 66,40 % | 0,2660 | - |

La configuration sélectionnée pour chaque cellule utilise la pondération racine des fréquences inverses, Adam, LR 0,003 et batch 256. Le modèle final choisi sur validation est WMC. Le classement observé sur le test sert à décrire les résultats ; aucune nouvelle sélection n'est faite avec ces scores de test.

Pour la démo, suivre **`output/gpu/COMMANDES_DEMO_GPU.txt`**. Les checkpoints sélectionnés et le vocabulaire sont dans `results/gpu_45min/`.
