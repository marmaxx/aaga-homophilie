**Projet : Revisiter les mesures d’homophilie**

*Présentation du projet:*

L’homophilie désigne la tendance des nœuds d’un graphe à être reliés à d’autres nœuds qui leur ressemblent. Plusieurs mesures permettent de quantifier cette propriété, mais elles présentent certaines limites, notamment lorsqu’il s’agit de comparer des graphes ayant des nombres de classes ou des répartitions de classes différents.

L’objectif de ce projet est d’implémenter les mesures d’homophilie présentées dans l’article Revisiting Graph Homophily Measures de Mikhail Mironov et Liudmila Prokhorenkova, puis de reproduire une partie des expériences proposées dans cet article sur de nouveaux jeux de données.

Le projet utilise notamment le jeu de données NCI1 afin de comparer les mesures et d’étudier leurs résultats.

*Installation et exécution:*

1. Installer les dépendances
   Python 3 doit être installé. Depuis un terminal ouvert dans le dossier du projet, exécuter :
   python3 -m pip install networkx pandas torch torch-geometric
2. Lancer le projet
   Exécuter les deux scripts dans l’ordre :
   python3 prepare_nci1.py
   python3 measures.py
   python3 agreement_matrix.py

Le premier script charge le jeu de données NCI1 et prépare les paires de graphes, le second calcule les mesures d’homophilie et le dernier génère la matrice des pourcentages d’accord entre elles.
