# analysepauses.py
import os
import subprocess
import re
import math
import cv2
import pandas as pd
import altair as alt
import shutil

# Importez ici les fonctions communes si besoin (vous pouvez les centraliser dans un module commun)
from analyseglobale import extraire_images_custom_ffmpeg, analyser_image, obtenir_transcription_whisper, get_transcript_text_at, detecter_pauses, exporter_concordancier_pause, analyser_audio_silence_detail

def afficher_pauses_contextes(pauses, images_data, transcript_segments, start_time, end_time, df_silence_detail, base_export_directory, emotion_data, extraction_interval):
    st.subheader("Analyse des Temps de Pause")
    if not pauses:
        st.write("Aucune pause > 1 s détectée.")
        return
    for pause in pauses:
        window_start = max(start_time, pause["start"] - 2)
        window_end = min(end_time, pause["end"] + 2)
        st.markdown(f"**Pause détectée** : de {pause['start']:.2f} s à {pause['end']:.2f} s (durée {pause['duration']:.2f} s)")
        st.write(f"Contexte de {window_start:.2f} s à {window_end:.2f} s")
        
        # Streamgraph de silence dans le contexte
        df_context = df_silence_detail[(df_silence_detail['Temps'] >= window_start) & (df_silence_detail['Temps'] <= window_end)]
        if not df_context.empty:
            sg_silence = alt.Chart(df_context).mark_area().encode(
                x=alt.X('Temps:Q', title='Temps (s)'),
                y=alt.Y('Temps_Silence:Q', title='Temps de silence (s)', stack='zero'),
                tooltip=['Temps', 'Temps_Silence']
            ).properties(
                title=f"Silence de {window_start:.2f} à {window_end:.2f} s",
                width=600,
                height=200
            )
            st.altair_chart(sg_silence, use_container_width=True)
        else:
            st.write("Pas de données audio pour ce contexte.")
        
        # Streamgraph des émotions dans le contexte (style global)
        df_context_emotions = pd.DataFrame([ed for ed in emotion_data if window_start <= ed["Seconde"] <= window_end])
        if not df_context_emotions.empty:
            df_melt = df_context_emotions.melt(id_vars=['Seconde'],
                                               value_vars=['angry','disgust','fear','happy','sad','surprise','neutral'],
                                               var_name='Emotion', value_name='Score')
            sg_emotions = alt.Chart(df_melt).mark_area().encode(
                x=alt.X('Seconde:Q', title='Seconde'),
                y=alt.Y('Score:Q', title='Score émotionnel', stack='center'),
                color=alt.Color('Emotion:N', scale=alt.Scale(scheme='category10')),
                tooltip=['Seconde', 'Emotion', 'Score']
            ).properties(
                title=f"Émotions de {window_start:.2f} à {window_end:.2f} s",
                width=600,
                height=200
            )
            st.altair_chart(sg_emotions, use_container_width=True)
        else:
            st.write("Pas de données d'émotions pour ce contexte.")
        
        # Grille d'images du contexte
        st.write("Grille d'images du contexte :")
        grid_cols = 5
        flat_images = []
        flat_captions = []
        t = math.floor(window_start / extraction_interval) * extraction_interval
        while t < window_end:
            caption_text = get_transcript_text_at(t, transcript_segments)
            sec_index = int(math.floor(t)) - start_time
            if sec_index < len(images_data) and images_data[sec_index]:
                img_path = images_data[sec_index][0]
                # Si t est dans la pause, border l'image
                if pause["start"] <= t < pause["end"]:
                    img = cv2.imread(img_path)
                    if img is not None:
                        bordered_img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0,0,255])
                        temp_dir = "temp"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, f"bordered_{t:.2f}.jpg")
                        cv2.imwrite(temp_path, bordered_img)
                        img_path = temp_path
                flat_images.append(img_path)
                flat_captions.append(f"{t:.2f} s - {caption_text}")
            t += extraction_interval
        for i in range(0, len(flat_images), grid_cols):
            cols = st.columns(grid_cols)
            for j, col in enumerate(cols):
                if i + j < len(flat_images):
                    col.image(flat_images[i+j], caption=flat_captions[i+j], use_container_width=True)
        
        # Export du concordancier pour ce contexte
        pause_dir, df_conc = exporter_concordancier_pause(pause, images_data, transcript_segments, start_time, end_time, base_export_directory)
        st.success(f"Export du contexte de pause réalisé dans : {pause_dir}")
        st.markdown("---")

def analyser_pauses(video_url, start_time, end_time, repertoire, extraction_interval, correction_synchro):
    # Téléchargement et extraction
    video_path = telecharger_video(video_url, repertoire)
    transcript_segments = obtenir_transcription_whisper(video_path, start_time, end_time, repertoire)
    # Pour réutiliser l'analyse globale, on peut imaginer que les images et autres données ont été calculées précédemment.
    # Ici, nous faisons un exemple simplifié :
    dossier_images = os.path.join(repertoire, "images_extraites")
    # Supposons que l'analyse globale ait déjà extrait ces données
    # Vous pouvez adapter ce bloc pour réutiliser des données stockées en session_state
    # Pour l'exemple, nous extrayons à nouveau pour chaque seconde (peut être optimisé)
    images_data = []
    for seconde in range(start_time, end_time + 1):
        imgs = extraire_images_custom_ffmpeg(video_path, dossier_images, seconde, extraction_interval)
        images_data.append(imgs)
    df_silence_detail = analyser_audio_silence_detail(video_path, start_time, end_time, extraction_interval, repertoire)
    pauses = detecter_pauses(video_path, start_time, end_time, repertoire)
    base_export_directory = os.path.join(repertoire, "pauses_export")
    os.makedirs(base_export_directory, exist_ok=True)
    # Pour l'émotion globale, vous pouvez simuler des valeurs (dans un cas réel, réutilisez vos données)
    emotion_data = []  # Remplacer par les résultats de l'analyse globale
    st.write("Analyse des temps de pause")
    afficher_pauses_contextes(pauses, images_data, transcript_segments, start_time, end_time, df_silence_detail, base_export_directory, emotion_data, extraction_interval)
