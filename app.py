import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="EduAnalytics AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
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
    .cluster-0 { background: linear-gradient(135deg, #534AB7 0%, #764ba2 100%); }
    .cluster-1 { background: linear-gradient(135deg, #0F6E56 0%, #38ef7d 100%); }
    .cluster-2 { background: linear-gradient(135deg, #993C1D 0%, #ff4b2b 100%); }
    .variance-box {
        background: #E6F1FB;
        border: 1px solid #85B7EB;
        border-radius: 10px;
        padding: 14px;
        margin: 10px 0;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        pca     = joblib.load('outputs/models/pca_model.pkl')
        scaler  = joblib.load('outputs/models/scaler_model.pkl')
        kmeans  = joblib.load('outputs/models/kmeans_model.pkl')
        metrics = joblib.load('outputs/models/metrics.pkl')
        df      = pd.read_csv('data/processed/students_clean.csv')
        return pca, scaler, kmeans, metrics, df
    except Exception as e:
        return None, None, None, None, None

pca, scaler, kmeans, metrics, df = load_assets()

st.title("🎓 EduAnalytics AI")
st.markdown("*Analyse intelligente des profils académiques par PCA + K-Means*")
st.markdown("---")

if pca is not None:
    var_pc1 = metrics['variance_pc1'] * 100
    var_pc2 = metrics['variance_pc2'] * 100
    sil     = metrics['silhouette']

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Dataset", "1 000 étudiants")
    col_m2.metric("Variance PC1", f"{var_pc1:.1f}%")
    col_m3.metric("Variance PC1+PC2", f"{var_pc1 + var_pc2:.1f}%",
                  help="Réponse question d'évaluation : variance conservée par les 2 premières composantes PCA")
    col_m4.metric("Score Silhouette", f"{sil:.4f}",
                  help="Qualité réelle du clustering K-Means (calculé sur les données)")

    st.markdown("---")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)
        st.header("Saisie des Scores")
        st.write("Ajustez les curseurs pour simuler un profil d'étudiant.")
        math    = st.slider("📐 Mathématiques", 0, 100, 70)
        reading = st.slider("📖 Lecture",        0, 100, 75)
        writing = st.slider("✍️ Écriture",       0, 100, 72)
        st.markdown("---")
        st.markdown(f"""
        <div class='variance-box'>
        <b>Résultat PCA</b><br>
        PC1 : <b>{var_pc1:.2f}%</b> de variance<br>
        PC2 : <b>{var_pc2:.2f}%</b> de variance<br>
        <b>Total : {var_pc1 + var_pc2:.2f}%</b> conservé
        </div>
        """, unsafe_allow_html=True)
        st.info(f"Score Silhouette K-Means : **{sil:.4f}**")

    input_df     = pd.DataFrame({'math score': [math], 'reading score': [reading], 'writing score': [writing]})
    scaled_input = scaler.transform(input_df)
    pca_input    = pca.transform(scaled_input)
    cluster      = kmeans.predict(pca_input)[0]

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🎯 Résultat de l'Analyse")
        cluster_names = ["Profil Équilibré", "Excellence Académique", "Besoin d'Accompagnement"]
        st.markdown(f"""
            <div class="prediction-card cluster-{cluster}">
                {cluster_names[cluster]}<br>
                <span style='font-size: 16px;'>Groupe ID : {cluster}</span>
            </div>
            """, unsafe_allow_html=True)

        score_global = (math + reading + writing) / 3
        m1, m2 = st.columns(2)
        m1.metric("Score Global", f"{score_global:.1f}/100")
        m2.metric("Score Silhouette", f"{sil:.4f}",
                  help="Métrique réelle calculée sur les données d'entraînement")

        with st.expander("💡 Recommandations Pédagogiques"):
            if cluster == 1:
                st.success("Félicitations ! Étudiant en excellence académique. Programmes d'approfondissement et tutorat par les pairs recommandés.")
            elif cluster == 0:
                st.info("Étudiant au profil équilibré. Maintenir le suivi actuel et encourager la participation extra-scolaire.")
            else:
                st.warning("Plan de soutien personnalisé recommandé en lecture et mathématiques dès le mois prochain.")

        with st.expander("📊 Coordonnées PCA de l'étudiant"):
            st.write(f"**PC1** (Performance globale) : `{pca_input[0][0]:.4f}`")
            st.write(f"**PC2** (Nuance du profil) : `{pca_input[0][1]:.4f}`")
            st.caption("Ces coordonnées sont utilisées par K-Means pour assigner le cluster.")

    with col2:
        st.subheader("📊 Cartographie PCA des étudiants")
        df_num    = df.select_dtypes(include=['number'])
        pca_all   = pca.transform(scaler.transform(df_num))
        clust_all = kmeans.predict(pca_all)

        plot_df = pd.DataFrame({
            'PC1': pca_all[:, 0],
            'PC2': pca_all[:, 1],
            'Cluster': [cluster_names[c] for c in clust_all]
        })

        fig = px.scatter(
            plot_df, x='PC1', y='PC2', color='Cluster',
            color_discrete_map={
                cluster_names[0]: '#534AB7',
                cluster_names[1]: '#0F6E56',
                cluster_names[2]: '#993C1D',
            },
            template="simple_white",
            labels={
                'PC1': f"PC1 — Performance globale ({var_pc1:.1f}% variance)",
                'PC2': f"PC2 — Nuance du profil ({var_pc2:.1f}% variance)"
            },
            opacity=0.5,
            title=f"Projection PCA — {var_pc1 + var_pc2:.1f}% de variance conservée"
        )
        fig.add_trace(go.Scatter(
            x=[pca_input[0][0]], y=[pca_input[0][1]],
            mode='markers',
            marker=dict(color='yellow', size=20, symbol='star', line=dict(width=2, color='black')),
            name='Étudiant Analysé'
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Analyse PCA — Réponse à la question d'évaluation")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        **Question :** *Quelle variance expliquée est conservée par les deux premières composantes PCA ?*

        **Réponse :**
        - **PC1** (Performance globale) : **{var_pc1:.2f}%** de la variance
        - **PC2** (Nuance du profil) : **{var_pc2:.2f}%** de la variance
        - **Total PC1 + PC2 : {var_pc1 + var_pc2:.2f}%** de la variance totale conservée

        > Les deux premières composantes PCA suffisent à représenter **{var_pc1 + var_pc2:.1f}%**
        > de l'information. Cette valeur élevée s'explique par la forte corrélation entre
        > les trois scores (math, lecture, écriture).
        """)
    with col_b:
        fig_var = go.Figure()
        fig_var.add_bar(
            x=['PC1', 'PC2'], y=[var_pc1, var_pc2],
            marker_color=['#185FA5', '#0F6E56'],
            text=[f"{var_pc1:.2f}%", f"{var_pc2:.2f}%"],
            textposition='outside'
        )
        fig_var.add_scatter(
            x=['PC1', 'PC2'], y=[var_pc1, var_pc1 + var_pc2],
            mode='lines+markers+text', name='Cumulé',
            line=dict(color='#993C1D', width=2), marker=dict(size=8),
            text=[f"{var_pc1:.1f}%", f"{var_pc1 + var_pc2:.1f}%"],
            textposition='top center'
        )
        fig_var.update_layout(
            title="Variance expliquée par composante",
            yaxis=dict(range=[0, 115], title="Variance (%)"),
            xaxis_title="Composante", template="simple_white",
            height=320, showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_var, use_container_width=True)

else:
    st.error("⚠️ Fichiers modèles manquants. Veuillez lancer `python main.py` dans votre terminal.")
    st.code("python main.py\nstreamlit run app.py")