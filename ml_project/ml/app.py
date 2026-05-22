"""
MediAI — Système Intelligent de Validation et de Diagnostic
Architecture : Modèle Qualité → filtre → Modèle Diagnostic
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="MediAI", page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
* { font-family: 'DM Sans', sans-serif !important; }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f4c81 100%);
    border: 1px solid #334155; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
}
.hero h1 { color:#fff; font-size:2.4rem; font-weight:700; margin:0; letter-spacing:-1px; }
.hero p  { color:#94a3b8; margin-top:.4rem; font-size:1.05rem; }
.accent  { color:#38bdf8; }

.step-card {
    border-radius:12px; padding:1.2rem 1.5rem; margin:.5rem 0;
    border-left:4px solid;
}
.step-valid   { background:#f0fdf4; border-color:#22c55e; }
.step-invalid { background:#fef2f2; border-color:#ef4444; }
.step-diag    { background:#eff6ff; border-color:#3b82f6; }

.result-erronee {
    background:#fef2f2; border:2px solid #ef4444;
    border-radius:14px; padding:1.8rem; text-align:center; margin:1rem 0;
}
.result-diabetique {
    background:#fff1f2; border:2px solid #f43f5e;
    border-radius:14px; padding:1.8rem; text-align:center; margin:1rem 0;
}
.result-sain {
    background:#f0fdf4; border:2px solid #22c55e;
    border-radius:14px; padding:1.8rem; text-align:center; margin:1rem 0;
}

.metric-box {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:10px; padding:1rem; text-align:center;
}
.metric-val  { font-size:1.9rem; font-weight:700; color:#0f172a; }
.metric-lbl  { font-size:.8rem; color:#64748b; margin-top:.2rem; }

.stButton>button {
    background:linear-gradient(135deg,#0ea5e9,#0284c7);
    color:white; border:none; border-radius:8px;
    padding:.65rem 2rem; font-weight:600; font-size:1rem;
    width:100%; transition:all .2s;
}
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(14,165,233,.35); }
</style>
""", unsafe_allow_html=True)

# ── Chargement des artefacts ───────────────────────────────────────
@st.cache_resource
def load_all():
    d = 'model_artifacts'
    return {
        'model_diag'    : joblib.load(f'{d}/model_diagnostic.pkl'),
        'scaler_diag'   : joblib.load(f'{d}/scaler_diagnostic.pkl'),
        'feat_diag'     : joblib.load(f'{d}/features_diagnostic.pkl'),
        'ns_diag'       : joblib.load(f'{d}/needs_scaling_diag.pkl'),
        'best_diag'     : joblib.load(f'{d}/best_diag_name.pkl'),
        'comp_diag'     : joblib.load(f'{d}/comp_diag.pkl'),

        'model_qual'    : joblib.load(f'{d}/model_qualite.pkl'),
        'scaler_qual'   : joblib.load(f'{d}/scaler_qualite.pkl'),
        'feat_qual'     : joblib.load(f'{d}/features_qualite.pkl'),
        'le_qualite'    : joblib.load(f'{d}/le_qualite.pkl'),
        'le_type'       : joblib.load(f'{d}/le_type.pkl'),
        'ns_qual'       : joblib.load(f'{d}/needs_scaling_qual.pkl'),
        'best_qual'     : joblib.load(f'{d}/best_qual_name.pkl'),
        'comp_qual'     : joblib.load(f'{d}/comp_qual.pkl'),

        'df_diab'       : pd.read_csv(f'{d}/diabetes_clean.csv'),
        'df_qual'       : pd.read_csv(f'{d}/qualite_labo.csv'),
    }

art = load_all()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 MediAI")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🔬 Diagnostic Complet",
        "📊 EDA — Diabète",
        "⚗️  EDA — Qualité Labo",
        "🤖 Comparaison Modèles",
        "📋 Architecture"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='background:#f1f5f9;border-radius:8px;padding:.9rem;border:1px solid #e2e8f0;'>
        <p style='color:#64748b;font-size:.75rem;margin:0;'>Modèle Diagnostic</p>
        <p style='color:#0f172a;font-weight:700;font-size:.95rem;margin:0;'>{art['best_diag']}</p>
        <p style='color:#64748b;font-size:.75rem;margin:.5rem 0 0;'>Modèle Qualité</p>
        <p style='color:#0f172a;font-weight:700;font-size:.95rem;margin:0;'>{art['best_qual']}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏥 Medi<span class="accent">AI</span></h1>
  <p>Système Intelligent de Validation et de Diagnostic d'Analyses Médicales</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 : DIAGNOSTIC COMPLET
# ══════════════════════════════════════════════════════════════════
if page == "🔬 Diagnostic Complet":

    st.markdown("## 🔬 Saisie des Paramètres")
    st.info("💡 Le système vérifie d'abord la qualité de l'analyse, puis lance le diagnostic uniquement si les conditions sont conformes.")

    # ── Bloc 1 : Qualité ──────────────────────────────────────────
    st.markdown("### ⚗️ Étape 1 — Conditions Pré-analytiques")
    c1, c2, c3 = st.columns(3)
    with c1:
        qualite      = st.selectbox("Qualité de l'échantillon", ["bon","moyen","mauvais"])
        type_prel    = st.selectbox("Type de prélèvement", ["veineux","capillaire","arteriel"])
    with c2:
        temperature  = st.slider("Température de stockage (°C)", 0.0, 37.0, 5.0, 0.5)
        delai        = st.slider("Délai avant analyse (heures)", 0.0, 24.0, 2.0, 0.5)
    with c3:
        centrifuge   = st.radio("Centrifugation effectuée ?", [1, 0],
                                format_func=lambda x: "Oui ✅" if x==1 else "Non ❌")
        hemolyse     = st.slider("Niveau d'hémolyse estimé", 0.0, 1.0, 0.05, 0.01)
        st.caption("0 = aucune | 1 = hémolyse totale")

    st.markdown("---")

    # ── Bloc 2 : Biologie ─────────────────────────────────────────
    st.markdown("### 🧬 Étape 2 — Paramètres Biologiques du Patient")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        pregnancies = st.number_input("Grossesses", 0, 17, 1)
        glucose     = st.number_input("Glucose (mg/dL)", 40, 200, 110)
    with d2:
        bp          = st.number_input("Pression artérielle (mmHg)", 40, 130, 72)
        skin        = st.number_input("Épaisseur cutanée (mm)", 0, 99, 20)
    with d3:
        insulin     = st.number_input("Insuline (μU/mL)", 0, 850, 80)
        bmi         = st.number_input("IMC (kg/m²)", 10.0, 70.0, 28.0, 0.1)
    with d4:
        dpf         = st.number_input("Antécédents familiaux (DPF)", 0.0, 2.5, 0.35, 0.01)
        age         = st.number_input("Âge (ans)", 21, 81, 35)
        st.caption("DPF = Diabetes Pedigree Function")

    st.markdown("---")
    _, btn_col, _ = st.columns([1,2,1])
    with btn_col:
        run = st.button("🚀 Lancer l'Analyse Complète", use_container_width=True)

    if run:
        # ── Prédiction Qualité ─────────────────────────────────
        q_enc = art['le_qualite'].transform([qualite])[0]
        t_enc = art['le_type'].transform([type_prel])[0]
        X_q   = np.array([[q_enc, temperature, delai, t_enc, centrifuge, hemolyse]])
        if art['ns_qual']:
            X_q = art['scaler_qual'].transform(X_q)

        qual_pred  = art['model_qual'].predict(X_q)[0]
        qual_proba = art['model_qual'].predict_proba(X_q)[0] if hasattr(art['model_qual'],'predict_proba') else None

        st.markdown("## 📋 Résultats")
        st.markdown("### Étape 1 — Vérification de la Qualité")

        qa_col, qb_col = st.columns(2)
        with qa_col:
            if qual_proba is not None:
                fig, ax = plt.subplots(figsize=(6,2.5))
                bars = ax.barh(['Valide','Erronée'], qual_proba,
                               color=['#22c55e','#ef4444'], height=0.4)
                for bar, v in zip(bars, qual_proba):
                    ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                            f'{v:.1%}', va='center', fontweight='bold')
                ax.set_xlim(0,1.15); ax.set_title("Probabilités — Qualité")
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                plt.tight_layout(); st.pyplot(fig)

        with qb_col:
            if qual_pred == 1:
                st.markdown("""
                <div class="result-erronee">
                    <h2>⚠️</h2>
                    <h3 style="color:#ef4444;">ANALYSE ERRONÉE</h3>
                    <p>Conditions pré-analytiques non conformes.<br>
                    <strong>Répétez le prélèvement</strong> avant tout diagnostic.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-sain" style="background:#f0fdf4;border-color:#22c55e;">
                    <h2>✅</h2>
                    <h3 style="color:#16a34a;">ANALYSE VALIDE</h3>
                    <p>Les conditions pré-analytiques sont conformes.<br>
                    Passage au diagnostic médical.</p>
                </div>""", unsafe_allow_html=True)

        # ── Prédiction Diagnostic (uniquement si valide) ───────
        if qual_pred == 0:
            st.markdown("### Étape 2 — Diagnostic Médical")

            X_d = np.array([[pregnancies, glucose, bp, skin,
                              insulin, bmi, dpf, age]])
            if art['ns_diag']:
                X_d = art['scaler_diag'].transform(X_d)

            diag_pred  = art['model_diag'].predict(X_d)[0]
            diag_proba = art['model_diag'].predict_proba(X_d)[0] if hasattr(art['model_diag'],'predict_proba') else None

            da_col, db_col = st.columns(2)
            with da_col:
                if diag_proba is not None:
                    fig, ax = plt.subplots(figsize=(6,2.5))
                    bars = ax.barh(['Non diabétique','Diabétique'], diag_proba,
                                   color=['#22c55e','#f43f5e'], height=0.4)
                    for bar, v in zip(bars, diag_proba):
                        ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                                f'{v:.1%}', va='center', fontweight='bold')
                    ax.set_xlim(0,1.15); ax.set_title("Probabilités — Diagnostic")
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    plt.tight_layout(); st.pyplot(fig)

            with db_col:
                if diag_pred == 1:
                    risk = diag_proba[1] if diag_proba is not None else 0.5
                    urgence = "⚡ Risque élevé — suivi urgent conseillé" if risk > 0.7 else "Consultation médicale recommandée"
                    st.markdown(f"""
                    <div class="result-diabetique">
                        <h2>🔴</h2>
                        <h3 style="color:#f43f5e;">DIABÉTIQUE</h3>
                        <p>{urgence}</p>
                        <p style="font-size:.85rem;color:#9f1239;">
                        Facteurs clés : Glucose élevé ({glucose} mg/dL), IMC ({bmi} kg/m²)
                        </p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-sain">
                        <h2>🟢</h2>
                        <h3 style="color:#16a34a;">NON DIABÉTIQUE</h3>
                        <p>Aucun signe de diabète détecté.</p>
                        <p style="font-size:.85rem;color:#166534;">
                        Glucose : {glucose} mg/dL | IMC : {bmi} kg/m²
                        </p>
                    </div>""", unsafe_allow_html=True)

            # ── Explication des facteurs ───────────────────────
            if diag_proba is not None and diag_proba[1] > 0.4:
                st.markdown("#### 🔍 Facteurs à surveiller")
                feat_names = ['Grossesses','Glucose','Pression art.','Épaisseur cut.',
                              'Insuline','IMC','Antécédents','Âge']
                feat_vals  = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
                norms      = ['-','70-100','60-80','10-30','16-166','18.5-24.9','-','21+']
                risks      = []
                for name, val, norm in zip(feat_names, feat_vals, norms):
                    flag = ""
                    if name == "Glucose" and val > 126: flag = "⚠️ Élevé"
                    elif name == "IMC" and val > 30:    flag = "⚠️ Obésité"
                    elif name == "Pression art." and val > 90: flag = "⚠️ Élevée"
                    if flag:
                        risks.append({'Paramètre':name,'Valeur':val,'Norme':norm,'Statut':flag})
                if risks:
                    st.dataframe(pd.DataFrame(risks), use_container_width=True, hide_index=True)
        else:
            st.warning("⛔ Le diagnostic est bloqué car l'analyse est erronée. Refaire le prélèvement.")

# ══════════════════════════════════════════════════════════════════
# PAGE 2 : EDA DIABÈTE
# ══════════════════════════════════════════════════════════════════
elif page == "📊 EDA — Diabète":
    st.markdown("## 📊 Exploration — Dataset Diabète (Pima Indians)")
    df = art['df_diab']

    m1,m2,m3,m4 = st.columns(4)
    for col, val, lbl in [
        (m1, len(df), "Patients"),
        (m2, df['Outcome'].sum(), "Diabétiques"),
        (m3, f"{df['Outcome'].mean():.1%}", "Taux diabète"),
        (m4, df.shape[1]-1, "Features"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Distributions","Boxplots","Corrélations"])

    FEAT = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']
    with tab1:
        feat_sel = st.selectbox("Feature", FEAT)
        fig, ax = plt.subplots(figsize=(10,4))
        for out, color, lbl in [(0,'#22c55e','Non diabétique'),(1,'#ef4444','Diabétique')]:
            ax.hist(df[df['Outcome']==out][feat_sel], bins=30, alpha=0.6,
                    color=color, label=lbl, edgecolor='none')
        ax.set_title(f"Distribution de {feat_sel}", fontweight='bold')
        ax.set_xlabel(feat_sel); ax.set_ylabel("Fréquence"); ax.legend()
        plt.tight_layout(); st.pyplot(fig)

    with tab2:
        fig, axes = plt.subplots(2,4,figsize=(18,8))
        for ax, feat in zip(axes.flat, FEAT):
            data = [df[df['Outcome']==0][feat], df[df['Outcome']==1][feat]]
            bp = ax.boxplot(data, labels=['Non Diab.','Diabétique'], patch_artist=True)
            for patch, c in zip(bp['boxes'],['#22c55e','#ef4444']):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_title(feat, fontweight='bold', fontsize=10)
        plt.tight_layout(); st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10,8))
        mask = np.triu(np.ones_like(df[FEAT+['Outcome']].corr(), dtype=bool))
        sns.heatmap(df[FEAT+['Outcome']].corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, ax=ax, linewidths=0.5, annot_kws={'size':9})
        ax.set_title("Matrice de Corrélation", fontweight='bold')
        plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 : EDA QUALITÉ
# ══════════════════════════════════════════════════════════════════
elif page == "⚗️  EDA — Qualité Labo":
    st.markdown("## ⚗️ Exploration — Dataset Qualité Labo (Semi-synthétique)")
    dq = art['df_qual']

    st.info("💡 Ce dataset a été simulé à partir de règles biologiques réelles (normes ISO 15189). Les données pré-analytiques réelles sont internes aux laboratoires et non publiées.")

    m1,m2,m3 = st.columns(3)
    for col, val, lbl in [
        (m1, len(dq), "Analyses"),
        (m2, dq['resultat_qualite'].sum(), "Erronées"),
        (m3, f"{dq['resultat_qualite'].mean():.1%}", "Taux erreur"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    fig, axes = plt.subplots(2,3,figsize=(16,9))
    fig.suptitle("Analyse du Dataset Qualité Pré-analytique", fontsize=13, fontweight='bold')

    vc = dq['resultat_qualite'].value_counts()
    axes[0,0].pie(vc.values, labels=['Valide','Erronée'],
                  autopct='%1.1f%%', colors=['#22c55e','#ef4444'],
                  wedgeprops={'edgecolor':'white','linewidth':2})
    axes[0,0].set_title("Répartition Valide/Erronée", fontweight='bold')

    ct = pd.crosstab(dq['qualite_echantillon'], dq['resultat_qualite'])
    ct.columns = ['Valide','Erronée']
    ct.plot(kind='bar', ax=axes[0,1], color=['#22c55e','#ef4444'], edgecolor='white', width=0.6)
    axes[0,1].set_title("Qualité → Résultat", fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=0)

    for res, color, lbl in [(0,'#22c55e','Valide'),(1,'#ef4444','Erronée')]:
        axes[0,2].hist(dq[dq['resultat_qualite']==res]['temperature_celsius'],
                       bins=25, alpha=0.6, color=color, label=lbl, edgecolor='none')
    axes[0,2].axvline(20, color='black', linestyle='--', linewidth=1.5, label='Seuil 20°C')
    axes[0,2].set_title("Température de Stockage", fontweight='bold')
    axes[0,2].set_xlabel("°C"); axes[0,2].legend(fontsize=8)

    for res, color, lbl in [(0,'#22c55e','Valide'),(1,'#ef4444','Erronée')]:
        axes[1,0].hist(dq[dq['resultat_qualite']==res]['delai_heures'],
                       bins=25, alpha=0.6, color=color, label=lbl, edgecolor='none')
    axes[1,0].axvline(8, color='black', linestyle='--', linewidth=1.5, label='Seuil 8h')
    axes[1,0].set_title("Délai avant Analyse (h)", fontweight='bold')
    axes[1,0].set_xlabel("Heures"); axes[1,0].legend(fontsize=8)

    for res, color, lbl in [(0,'#22c55e','Valide'),(1,'#ef4444','Erronée')]:
        axes[1,1].hist(dq[dq['resultat_qualite']==res]['niveau_hemolyse'],
                       bins=25, alpha=0.6, color=color, label=lbl, edgecolor='none')
    axes[1,1].axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Seuil 0.5')
    axes[1,1].set_title("Niveau d'Hémolyse", fontweight='bold')
    axes[1,1].legend(fontsize=8)

    ct2 = pd.crosstab(dq['centrifuge'], dq['resultat_qualite'])
    ct2.index = ['Non centrifugé','Centrifugé']
    ct2.columns = ['Valide','Erronée']
    ct2.plot(kind='bar', ax=axes[1,2], color=['#22c55e','#ef4444'], edgecolor='white', width=0.5)
    axes[1,2].set_title("Centrifugation → Résultat", fontweight='bold')
    axes[1,2].tick_params(axis='x', rotation=0)

    plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 : COMPARAISON MODÈLES
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Comparaison Modèles":
    st.markdown("## 🤖 Comparaison des Algorithmes")

    tab_diag, tab_qual = st.tabs(["Modèle 1 — Diagnostic Diabète", "Modèle 2 — Qualité Labo"])

    for tab, comp, best_name, titre in [
        (tab_diag, art['comp_diag'], art['best_diag'], "Diagnostic"),
        (tab_qual, art['comp_qual'], art['best_qual'], "Qualité Labo"),
    ]:
        with tab:
            st.success(f"✅ **Meilleur modèle ({titre}) : {best_name}**")
            display = comp.copy()
            display.columns = [c.replace(' Weighted','') for c in display.columns]
            st.dataframe(display.style.format("{:.2%}").highlight_max(axis=0, color='#dcfce7'),
                         use_container_width=True)

            fig, axes = plt.subplots(1,2,figsize=(14,5))
            fig.suptitle(f"Comparaison — Modèle {titre}", fontsize=13, fontweight='bold')
            models = comp.index.tolist()
            colors = ['#22c55e' if m==best_name else '#cbd5e1' for m in models]

            col_f1 = 'F1 Weighted' if 'F1 Weighted' in comp.columns else comp.columns[3]
            axes[0].barh(models, comp[col_f1], color=colors, height=0.5, edgecolor='white')
            for i, v in enumerate(comp[col_f1]):
                axes[0].text(v+0.005, i, f'{v:.2%}', va='center', fontweight='bold')
            axes[0].set_xlim(0,1.1); axes[0].set_title("F1 Score", fontweight='bold')
            axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

            x = np.arange(len(models)); w = 0.2
            palette = ['#3b82f6','#22c55e','#f59e0b','#ef4444']
            for i, (col, c) in enumerate(zip(comp.columns[:4], palette)):
                axes[1].bar(x+i*w, comp[col], w, label=col, color=c, alpha=0.85, edgecolor='white')
            axes[1].set_xticks(x+w*1.5)
            axes[1].set_xticklabels([m.replace(' ','\n') for m in models], fontsize=9)
            axes[1].set_ylim(0,1.15); axes[1].legend(fontsize=8)
            axes[1].set_title("Multi-Métriques", fontweight='bold')
            axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

            plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 : ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
elif page == "📋 Architecture":
    st.markdown("## 📋 Architecture du Système")
    st.markdown("""
    ### 🏗️ Pourquoi deux modèles séparés ?

    Un résultat médical passe par **deux étapes distinctes** avant d'être fiable :

    | Étape | Type | Rôle |
    |---|---|---|
    | **Qualité pré-analytique** | Logistique / Labo | Valider les conditions du prélèvement |
    | **Diagnostic médical** | Médecine clinique | Interpréter les marqueurs biologiques |

    Mélanger ces deux dimensions donne un modèle artificiellement parfait qui apprend des règles triviales (ex: *qualité=mauvais → erronée*) sans rien apprendre de médical.

    ### 🔁 Logique de décision

    ```
    Entrée patient
         │
         ▼
    ┌────────────────────────┐
    │  Modèle 2 : Qualité    │
    │  (délai, temp, ...)    │
    └────────────────────────┘
         │
    ┌────┴────────────────┐
    │                     │
    Erronée (1)         Valide (0)
    │                     │
    ▼                     ▼
    ⛔ BLOQUER          ┌─────────────────────┐
    "Refaire le         │  Modèle 1 :          │
    prélèvement"        │  Diagnostic Diabète  │
                        └─────────────────────┘
                             │          │
                        Diabétique   Non diabétique
                             │          │
                            🔴         🟢
    ```

    ### 🧬 Datasets utilisés

    **Modèle Diagnostic — Dataset réel**
    - Source : Pima Indians Diabetes Dataset (Kaggle / UCI)
    - 768 patientes, 8 features biologiques
    - Cible : 0 = Non diabétique | 1 = Diabétique

    **Modèle Qualité — Dataset semi-synthétique**
    - Justification : *"Les données pré-analytiques sont internes aux laboratoires médicaux et ne sont pas publiées. Ce dataset a été simulé selon les normes ISO 15189 de qualité en laboratoire médical."*
    - 800 analyses, 6 variables de condition
    - Cible : 0 = Valide | 1 = Erronée

    ### 🔑 Choix des métriques
    - **F1 Score pondéré** → dataset déséquilibré (l'accuracy seule est trompeuse)
    - **Rappel (diabétiques)** → un Faux Négatif (diabétique non détecté) est plus grave qu'un Faux Positif
    - **Cross-validation 5-fold** → stabilité des performances, pas de chance sur le split
    """)
