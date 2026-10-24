# Guide du projet final

Le projet final utilise `results/gpu_45min/` : 15 000 avis train, 1 500 validation, 3 000 test et huit époques pour chacun des 12 réglages. L'entraînement a pris 37 min 49 s.

## Tester

Depuis le terminal dans `projet_rnn`, avec l'environnement Conda du dossier parent :

```powershell
..\.venv\python.exe -m unittest discover -s tests -v
..\.venv\python.exe -m src.predict --checkpoint results/gpu_45min/checkpoints/wmc_class_weight.pt --text "This medicine helped me sleep but caused nausea."
```

Les cinq tests doivent afficher OK. Les commandes pour les trois modèles sont dans `COMMANDES_DEMO_GPU.txt`. Pas besoin de refaire l'entraînement.

## Fichiers à utiliser

- `README.md` : instructions et protocole final.
- `output/pdf/report.pdf` : rapport final de cinq pages en anglais.
- `output/pdf/report.md` : texte éditable du rapport.
- `output/gpu/hyperparameters.csv` : les 12 expériences sur validation.
- `output/gpu/figures/` : graphiques et matrices de confusion.
- `output/gpu/Projet_RNN_GPU.zip` : archive à partager, avec installation pour un ami.

Le rapport vient des mesures enregistrées. Il se reconstruit avec `tools/build_report.py`, sans entraînement. Le PowerPoint est retiré ; tu le prépareras plus tard.

## Avant la remise

Ajouter les noms des 5 ou 6 membres du même TP, le groupe et l'URL du dépôt GitHub public dans `report_details.json`, puis relancer `tools/build_report.py`. Le dépôt n'a pas encore été publié et le formulaire du professeur n'a pas été envoyé.

Les notebooks Colab et les résultats préliminaires ont été retirés. Utiliser `src/train_fast.py` et `results/gpu_45min/`. Les trois checkpoints sélectionnés sont inclus dans Git ; les données, environnements, archives ZIP et autres checkpoints sont exclus.
