# analysedebit.py
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

def analyser_debit(video_url, start_time, end_time, repertoire):
    # Pour cet exemple, nous réutilisons la transcription via Whisper (vous pouvez la stocker en session_state dans main.py)
    from analyseglobale import telecharger_video, obtenir_transcription_whisper
    video_path = telecharger_video(video_url, repertoire)
    transcript_segments = obtenir_transcription_whisper(video_path, start_time, end_time, repertoire)
    
    def calculer_debit(transcript_segments):
        debits = []
        for segment in transcript_segments:
            start = float(segment['start'])
            end = float(segment.get('end', start + segment.get('duration', 0)))
            texte = segment['text']
            nb_mots = len(texte.split())
            duree = end - start if end - start > 0 else 1
            debit = (nb_mots / duree) * 60  # mots par minute
            debits.append({
                'start': start,
                'end': end,
                'nb_mots': nb_mots,
                'duree': round(duree, 2),
                'debit_mpm': round(debit, 2)
            })
        return debits
    
    debits = calculer_debit(transcript_segments)
    df_debit = pd.DataFrame(debits)
    st.subheader("Débit de parole par segment")
    st.dataframe(df_debit)
    
    chart = alt.Chart(df_debit).mark_bar().encode(
        x=alt.X('start:Q', title='Début du segment (s)'),
        y=alt.Y('debit_mpm:Q', title='Débit de parole (mots/min)'),
        tooltip=['start', 'end', 'nb_mots', 'debit_mpm']
    ).properties(
        title="Débit de parole par segment",
        width=600,
        height=300
    )
    st.altair_chart(chart, use_container_width=True)
