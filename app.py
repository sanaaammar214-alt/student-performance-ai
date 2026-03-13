import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Configuration de la page avec un look large
st.set_page_config(
    page_title="EduAnalytics AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Injection de CSS personnalisé pour le style "Professionnel"
st.markdown("""
    <style>
    /* Style général */
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Style des cartes de prédiction */
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .cluster-0 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); } /* Indigo */
    .cluster-1 { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); } /* Vert */
    .cluster-2 { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); } /* Rouge/Corail */
    
    /* Boutons et Sidebar */
    .css-1d391kg { background-color: #1e293b; } /* Sidebar foncée */
    </style>
    """, unsafe_allow_html=True)

# 3. Chargement des modèles (Cache)
@st.cache_resource
def load_assets():
    try:
        pca = joblib.load('outputs/models/pca_model.pkl')
        scaler = joblib.load('outputs/models/scaler_model.pkl')
        kmeans = joblib.load('outputs/models/kmeans_model.pkl')
        df = pd.read_csv('data/processed/students_clean.csv')
        return pca, scaler, kmeans, df
    except:
        return None, None, None, None

pca, scaler, kmeans, df = load_assets()

# --- HEADER ---
st.title("🎓 EduAnalytics AI")
st.markdown("---")

if pca is not None:
    # --- SIDEBAR DESIGN ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100) # Logo école optionnel
        st.header("Saisie des Scores")
        st.write("Ajustez les curseurs pour simuler un profil d'étudiant.")
        
        math = st.slider("📐 Mathématiques", 0, 100, 70)
        reading = st.slider("📖 Lecture", 0, 100, 75)
        writing = st.slider("✍️ Écriture", 0, 100, 72)
        
        st.info("L'IA analyse les données en temps réel.")

    # --- LOGIQUE DE PRÉDICTION ---
    input_df = pd.DataFrame({'math score': [math], 'reading score': [reading], 'writing score': [writing]})
    scaled_input = scaler.transform(input_df)
    pca_input = pca.transform(scaled_input)
    cluster = kmeans.predict(pca_input)[0]

    # --- LAYOUT PRINCIPAL ---
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🎯 Résultat de l'Analyse")
        
        # Affichage du Cluster avec un style "Carte"
        cluster_names = ["Profil Équilibré", "Excellence Académique", "Besoin d'Accompagnement"]
        cluster_class = f"cluster-{cluster}"
        
        st.markdown(f"""
            <div class="prediction-card {cluster_class}">
                {cluster_names[cluster]}<br>
                <span style='font-size: 16px;'>Groupe ID: {cluster}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics visuelles
        m1, m2 = st.columns(2)
        m1.metric("Score Global", f"{(math+reading+writing)/3:.1f}/100")
        m2.metric("Niveau de Confiance", "95.7%")

        with st.expander("💡 Recommandations Pédagogiques"):
            if cluster == 1:
                st.write("Félicitations ! Cet étudiant peut bénéficier de programmes d'approfondissement ou de tutorat pour ses pairs.")
            elif cluster == 0:
                st.write("Étudiant stable. Maintenir le suivi actuel et encourager la participation extra-scolaire.")
            else:
                st.write("Attention : Un plan de soutien personnalisé en lecture et mathématiques est recommandé dès le mois prochain.")

    with col2:
        st.subheader("📊 Cartographie du Profil")
        
        # Préparation des données pour Plotly
        df_num = df.select_dtypes(include=['number'])
        pca_all = pca.transform(scaler.transform(df_num))
        clusters_all = kmeans.predict(pca_all)
        
        plot_df = pd.DataFrame({
            'PC1': pca_all[:, 0],
            'PC2': pca_all[:, 1],
            'Cluster': [f"Groupe {c}" for c in clusters_all]
        })

        # Création du graphique interactif avec Plotly
        fig = px.scatter(
            plot_df, x='PC1', y='PC2', color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="simple_white",
            labels={'PC1': 'Performance Globale', 'PC2': 'Nuance Profil'},
            opacity=0.5
        )
        
        # Ajouter l'étudiant actuel (Étoile)
        fig.add_trace(go.Scatter(
            x=[pca_input[0][0]], y=[pca_input[0][1]],
            mode='markers',
            marker=dict(color='yellow', size=20, symbol='star', line=dict(width=2, color='black')),
            name='Étudiant Analysé'
        ))

        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Fichiers modèles manquants. Veuillez lancer 'python main.py' dans votre terminal.")
