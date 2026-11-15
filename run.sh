#!/usr/bin/env bash
set -e

# Détection de l'interpréteur Python (environnement virtuel local si présent, sinon python3)
if [ -d "env" ] && [ -f "env/bin/python" ]; then
    PYTHON_CMD="env/bin/python"
elif [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "Lancement des expériences d'homophilie..."

# 1. Calcul des matrices d'accord (Section 3 & Annexe B)
echo ""
echo "[1/2] Calcul des matrices d'accord..."
$PYTHON_CMD agreement_matrix.py

# 2. Calcul du Tableau 5 comparatif (Section 6)
echo ""
echo "[2/2] Calcul du tableau comparatif des moyennes..."
$PYTHON_CMD table_5_comparison.py

echo ""
echo "Terminé ! Fichiers générés dans results/ :"
ls results/