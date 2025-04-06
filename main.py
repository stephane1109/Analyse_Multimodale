# main.py
import streamlit as st
from analyseglobale import analyser_globale
from analysepauses import analyser_pauses
from analysedebit import analyser_debit

st.title("Application d'Analyse des Émotions dans une Vidéo")
st.markdown("Bienvenue sur l'application d'analyse des émotions.")

# Paramètres dans la barre latérale
st.sidebar.header("Paramètres")
repertoire = st.sidebar.text_input("Chemin du répertoire de travail", "")
video_url = st.sidebar.text_input("URL de la vidéo à analyser", "")
start_time = st.sidebar.number_input("Temps de départ (en secondes)", min_value=0.0, value=0.0, step=0.1)
end_time = st.sidebar.number_input("Temps d'arrivée (en secondes)", min_value=start_time, value=start_time+10.0, step=0.1)
# Pour 25 fps, extraction_interval est fixe à 0.04 s
extraction_interval = 0.04
correction_synchro = st.sidebar.number_input("Correction de synchronisation (en secondes)", value=0.0, step=0.01)

if st.sidebar.button("Lancer l'analyse"):
    if video_url and repertoire:
        st.session_state.video_url = video_url
        st.session_state.repertoire = repertoire
        st.session_state.start_time = start_time
        st.session_state.end_time = end_time
        st.session_state.extraction_interval = extraction_interval
        st.session_state.correction_synchro = correction_synchro
        st.experimental_rerun()
    else:
        st.sidebar.error("Veuillez définir le répertoire et l'URL de la vidéo.")

if "video_url" in st.session_state:
    tabs = st.tabs(["Analyse Globale", "Analyse des Temps de Pause", "Analyse du Débit de Parole"])
    
    with tabs[0]:
        st.header("Analyse Globale")
        analyser_globale(
            st.session_state.video_url,
            st.session_state.start_time,
            st.session_state.end_time,
            st.session_state.repertoire,
            st.session_state.extraction_interval,
            st.session_state.correction_synchro
        )
    
    with tabs[1]:
        st.header("Analyse des Temps de Pause")
        analyser_pauses(
            st.session_state.video_url,
            st.session_state.start_time,
            st.session_state.end_time,
            st.session_state.repertoire,
            st.session_state.extraction_interval,
            st.session_state.correction_synchro
        )
    
    with tabs[2]:
        st.header("Analyse du Débit de Parole")
        analyser_debit(
            st.session_state.video_url,
            st.session_state.start_time,
            st.session_state.end_time,
            st.session_state.repertoire
        )
