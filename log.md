# Log des runs

Toutes les runs : modèle `sao10k/l3-lunaris-8b`, 100 questions/condition (seed 1312), max_tokens 500, juge = même modèle.
Accuracies = proportion de jugements vrais par condition.

## Run 1 — full, sans ritalin, temp 0.2
- Fichier : `results/output.csv` (05:15)
- Note : run initiale, le modèle n'avait pas encore ritalin dans CONDITIONS
- Acc : control 0.45 · lsd 0.07 · cocaine 0.11 · alcohol 0.08 · cannabis 0.21

## Run 2 — full prompt avec ritalin, temp 0.2
- Fichier : `results/output_2026-08-10_12-04-01.csv`
- Acc : control 0.38 · lsd 0.08 · cocaine 0.10 · alcohol 0.11 · cannabis 0.24 · ritalin 0.37

## Run 3 — prompt simple (plain, mot seul), temp 0.2
- Fichier : `results/output_2026-08-10_12-52-33.csv`
- Acc : control 0.46 · lsd 0.03 · cocaine 0.06 · alcohol 0.07 · cannabis 0.26 · ritalin 0.35
- Note : le MOT SEUL suffit à dégrader (amorçage sémantique, pas le cadrage descriptif)

## Run 4 — full prompt avec ritalin, temp 0.2
- Fichier : `results/output_2026-08-10_14-32-53.csv`
- Acc : control 0.48 · lsd 0.16 · cocaine 0.11 · alcohol 0.11 · cannabis 0.35 · ritalin 0.51

## Run 5 — full, temp 0.8
- Fichier : `results/output_2026-08-13_22-11-07.csv`
- Acc : control 0.51 · lsd 0.16 · cocaine 0.11 · alcohol 0.10 · cannabis 0.42 · ritalin 0.53
- Note : la hausse de température remonte tout le profil mais ne change pas le classement (lsd/cocaine/alcohol restent écrasés)

## Run 6 — full, temp 1.5
- Fichier : `results/output_2026-08-13_23-52-25.csv`
- Acc : control 0.20 · lsd 0.02 · cocaine 0.01 · alcohol 0.02 · cannabis 0.03 · ritalin 0.12
- Note : effondrement GLOBAL — même control passe sous 0.2, toutes les substances ≈ 0. La température 1.5 détruit la cohérence (réponses moyennes : 250 mots vs 168 à temp 0.2, +48%) — le juge ne retrouve plus la vérité dans le bruit. L'effet drogue devient indiscernable car tout est au plancher
