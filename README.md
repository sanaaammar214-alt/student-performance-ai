# 🎓 EduAnalytics AI - Performance des Étudiants

Ce projet utilise l'apprentissage non-supervisé (K-Means) et la réduction de dimension (PCA) pour segmenter et analyser les performances académiques.

## 📊 Résultats Clés
L'analyse a révélé des insights majeurs sur le comportement des étudiants :

- **Réduction de dimension (PCA)** : Les deux premières composantes principales capturent **98.50%** de la variance totale des données. Cela permet de visualiser les profils avec une perte d'information quasi-nulle.
- **Clustering (K-Means)** : 
  - **Nombre de clusters (k)** : Nous avons retenu **k=3** pour sa pertinence métier (Profil Excellence, Équilibré, et Besoin de Soutien).
  - **Score Silhouette** : L'indice obtenu est de **0.4247**, confirmant une segmentation robuste et cohérente.

## 🚀 Utilisation
1. **Activer l'environnement virtuel** :
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. **Installer les dépendances** :
   ```bash
   pip install -r projet_notes_etudiants/requirements.txt
   ```
3. **Lancer le pipeline d'entraînement** (génère les modèles et metrics) :
   ```bash
   python projet_notes_etudiants/main.py
   ```
4. **Lancer l'application interactive (Streamlit)** :
   ```bash
   streamlit run projet_notes_etudiants/app.py
   ```

## 🛠 Structure du projet
- `src/` : Modules Python pour le preprocessing, PCA, et clustering.
- `app.py` : Dashboard interactif avec Plotly et CSS personnalisé.
- `main.py` : Script principal pour exécuter le pipeline de données.
- `data/` : Données brutes et transformées.
- `outputs/` : Modèles sérialisés (`.pkl`) et graphiques.
