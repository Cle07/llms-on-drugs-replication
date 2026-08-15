✔️ Mesurer variation selon température
✔️ Mesurer longueur réponse
✔️ Retirer les prefix et donner que la substance
⭕ **Faire varier la dose du psychotrope dans le prompt** (ex. faible / modérée / forte pour chaque substance) — tester si l'augmentation de dose dégrade davantage la perf, et si la dégradation est monotone ou plate au-delà d'un seuil — cf. papier LLMs on Drugs qui discute l'effet dose-dépendant
⭕ Calculer embeddings ou autre méthode stylométrique pour clusteriser les réponses et voir si variations dans le sens/style des réponses selon les drogues ?
✔️ **Ajouter un groupe control ET un groupe sobre distincts** — tester si le prompt « tu es sobre » (persona sobre) change la perf par rapport à poser la question directement sans aucune persona (baseline pure). Isoler l'effet du cadrage « sobre » lui-même, indépendamment de la drogue
⭕ Faire une run avec un modèle juge différent
