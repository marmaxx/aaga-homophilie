# Projet : Revisiting Graph Homophily Measures

Ce projet implémente et évalue les mesures d'homophilie de graphes présentées dans l'article :
> **Mikhail Mironov and Liudmila Prokhorenkova**, *Revisiting Graph Homophily Measures*, Proceedings of the Third Learning on Graphs Conference (LoG 2024), PMLR 269.

---

## 1. Structure du projet

* **`measures.py`** : Définition des 6 mesures d'homophilie :
  * $h_{edge}$ (`edge_homophily`)
  * $h_{node}$ (`node_homophily`)
  * $h_{class}$ (`class_homophily`)
  * $h_{adj}$ (`adjusted_homophily`)
  * $h_{unb}$ (`unbiased_homophily`, formule 4 rapide, $\alpha = 0$)
  * $h_{unb}^\alpha$ (`unbiased_homophily_alpha`, formule 2 générale)
* **`agreement_matrix.py`** : Chargement des graphes, gestion des paires et calcul des matrices d'accord complètes dans le répertoire `results/`.
* **`table_5_comparison.py`** : Calcul du tableau comparatif des moyennes d'homophilie (Section 6, Tableau 5) dans `results/`.

---

## 2. Installation et exécution

### Dépendances requises
```bash
pip install networkx pandas numpy torch torch-geometric
```

### Exécution rapide (tout en un)
```bash
./run.sh
```

### Exécution détaillée des scripts

1. **Calcul des matrices d'accord complètes (Section 3 & Annexe B) :**
   ```bash
   python3 agreement_matrix.py
   ```
   Ce script traite automatiquement **NCI1**, **MUTAG**, **PROTEINS** et **Synthetic SBM**, puis enregistre les matrices dans `results/` :
   * `results/agreement_matrix_nci1.csv`
   * `results/agreement_matrix_mutag.csv`
   * `results/agreement_matrix_proteins.csv`
   * `results/agreement_matrix_synthetic.csv`

2. **Génération du Tableau 5 comparatif (Section 6) :**
   ```bash
   python3 table_5_comparison.py
   ```
   Génère `results/table_5.csv` avec les moyennes et écart-types sur l'ensemble des jeux de données.
