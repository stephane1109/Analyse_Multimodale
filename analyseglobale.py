# analyseglobale.py
import os
import subprocess
import numpy as np
import cv2
from yt_dlp import YoutubeDL
import altair as alt
import pandas as pd
import shutil
import re
import math
import whisper
from fer import FER

# Fonctions internes réutilisées (vous pouvez extraire ces fonctions communes dans un module séparé si besoin)
def telecharger_video(url, repertoire):
    video_path = os.path.join(repertoire, 'video.mp4')
    if os.path.exists(video_path):
        return video_path
    ydl_opts = {'outtmpl': video_path, 'format': 'best'}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return video_path

def obtenir_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def extraire_images_custom_ffmpeg(video_path, repertoire, seconde, extraction_interval):
    images_extraites = []
    nb_images = int(1 / extraction_interval)
    for frame in range(nb_images):
        image_path = os.path.join(repertoire, f"image_{nb_images}fps_{seconde}_{frame}.jpg")
        if os.path.exists(image_path):
            images_extraites.append(image_path)
            continue
        temps = seconde + frame * extraction_interval
        cmd = ['ffmpeg', '-ss', str(temps), '-i', video_path, '-frames:v', '1', '-q:v', '2', image_path, '-y']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            break
        images_extraites.append(image_path)
    return images_extraites

def analyser_image(image_path, detector):
    image = cv2.imread(image_path)
    if image is None:
        return {}
    resultats = detector.detect_emotions(image)
    if resultats:
        for result in resultats:
            (x, y, w, h) = result["box"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for idx, (emotion, score) in enumerate(result['emotions'].items()):
                texte = f"{emotion}: {score:.2f}"
                cv2.putText(image, texte, (x, y + h + 20 + idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(image_path, image)
        return {k: round(v, 2) for k, v in resultats[0]['emotions'].items()}
    return {}

def emotion_dominante_par_moyenne(emotions_list):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    moyenne_emotions = {e: round(np.mean([emo.get(e, 0) for emo in emotions_list]), 2) for e in emotions}
    emotion_dominante = max(moyenne_emotions, key=moyenne_emotions.get)
    return moyenne_emotions, emotion_dominante

def obtenir_transcription_whisper(video_path, start_time, end_time, repertoire):
    audio_path = os.path.join(repertoire, "temp_audio.wav")
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ss', str(start_time), '-to', str(end_time),
           '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    segments = result.get("segments", [])
    os.remove(audio_path)
    return segments

def get_transcript_text_at(timestamp, transcript_segments, correction=0.0):
    ts = timestamp + correction
    for segment in transcript_segments:
        start = float(segment['start'])
        duration = float(segment.get('duration', 0))
        if start <= ts < start + duration:
            return segment['text']
    # Si aucun segment ne correspond exactement, retourne le texte du segment le plus proche
    closest_segment = None
    min_diff = float('inf')
    for segment in transcript_segments:
        diff = abs(ts - float(segment['start']))
        if diff < min_diff:
            min_diff = diff
            closest_segment = segment
    return closest_segment['text'] if closest_segment else ""

def creer_concordancier(images_data, emotions_data, transcript_segments, repertoire, start_time, extraction_interval, pauses):
    data = {'Timestamp': [], 'Image': [], 'Emotions': [], 'Transcription': []}
    for sec_index, images_list in enumerate(images_data):
        base_time = start_time + sec_index
        for idx, img_path in enumerate(images_list):
            timestamp = base_time + idx * extraction_interval
            texte = get_transcript_text_at(timestamp, transcript_segments)
            emotions = emotions_data[sec_index] if sec_index < len(emotions_data) else {}
            data['Timestamp'].append(timestamp)
            data['Image'].append(os.path.basename(img_path))
            data['Emotions'].append(emotions)
            data['Transcription'].append(texte)
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(repertoire, "concordancier_emotions.xlsx"), index=False)
    df.to_csv(os.path.join(repertoire, "concordancier_emotions.csv"), index=False)
    st.write("Concordancier global généré et exporté.")

def analyser_audio_silence(video_path, start_time, end_time, repertoire):
    audio_path = os.path.join(repertoire, "audio_segment.wav")
    cmd_extract = ['ffmpeg', '-y', '-i', video_path, '-ss', str(start_time), '-to', str(end_time),
                   '-vn', '-acodec', 'pcm_s16le', audio_path]
    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_silence = ['ffmpeg', '-i', audio_path, '-af', 'silencedetect=noise=-30dB:d=0.5', '-f', 'null', '-']
    result = subprocess.run(cmd_silence, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stderr.decode('utf-8')
    matches = re.findall(r"silence_duration: ([0-9.]+)", output)
    durations = [float(x) for x in matches]
    total_silence = round(sum(durations), 2)
    total_duration = end_time - start_time
    temps_parole = round(total_duration - total_silence, 2)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return {'temps_parole': temps_parole, 'temps_silence': total_silence}

def analyser_audio_silence_detail(video_path, start_time, end_time, extraction_interval, repertoire):
    audio_path = os.path.join(repertoire, "audio_segment.wav")
    cmd_extract = ['ffmpeg', '-y', '-i', video_path, '-ss', str(start_time), '-to', str(end_time),
                   '-vn', '-acodec', 'pcm_s16le', audio_path]
    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_silence = ['ffmpeg', '-i', audio_path, '-af', 'silencedetect=noise=-30dB:d=0.5', '-f', 'null', '-']
    result = subprocess.run(cmd_silence, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stderr.decode('utf-8')
    matches = re.findall(r"silence_start: ([0-9.]+).*?silence_duration: ([0-9.]+)", output, re.DOTALL)
    num_bins = int(np.ceil((end_time - start_time) / extraction_interval))
    bins = {start_time + i * extraction_interval: 0.0 for i in range(num_bins)}
    for start_str, dur_str in matches:
        t_start = float(start_str)
        t_dur = float(dur_str)
        if start_time <= t_start < end_time:
            bin_index = int(math.floor((t_start - start_time) / extraction_interval))
            bin_key = start_time + bin_index * extraction_interval
            bins[bin_key] += t_dur
    df = pd.DataFrame({'Temps': list(bins.keys()), 'Temps_Silence': [round(val, 2) for val in bins.values()]})
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return df

def afficher_dataframe_et_streamgraph(df_emotions, df_emotion_global, start_time, end_time, repertoire):
    st.subheader("Scores des émotions par frame (global)")
    st.dataframe(df_emotions)
    df_emotions['Frame_Index'] = df_emotions['Frame'].apply(lambda f: int(f.split('_')[1]))
    df_melt = df_emotions.melt(id_vars=['Frame_Index', 'Seconde'],
                               value_vars=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                               var_name='Emotion', value_name='Score')
    sg_frames = alt.Chart(df_melt).mark_area().encode(
        x=alt.X('Frame_Index:Q', title='Index de Frame'),
        y=alt.Y('Score:Q', title='Score des émotions', stack='center'),
        color=alt.Color('Emotion:N', scale=alt.Scale(scheme='category10')),
        tooltip=['Frame_Index', 'Emotion', 'Score']
    ).properties(title='Streamgraph des émotions par frame (global)', width=800, height=400)
    st.altair_chart(sg_frames, use_container_width=True)
    
    st.write("Moyenne des émotions par seconde (global)")
    st.dataframe(df_emotion_global)
    df_sec = df_emotion_global.melt(id_vars=['Seconde'],
                                    value_vars=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                                    var_name='Emotion', value_name='Score')
    sg_seconds = alt.Chart(df_sec).mark_area().encode(
        x=alt.X('Seconde:Q', title=f'Secondes (de {start_time} à {end_time})'),
        y=alt.Y('Score:Q', title='Score des émotions', stack='center'),
        color=alt.Color('Emotion:N', scale=alt.Scale(scheme='category10')),
        tooltip=['Seconde', 'Emotion', 'Score']
    ).properties(title='Streamgraph des moyennes des émotions par seconde (global)', width=800, height=400)
    st.altair_chart(sg_seconds, use_container_width=True)
    
    st.subheader("Analyse globale des émotions")
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    stats = []
    for e in emotions:
        scores = df_emotion_global[e].tolist()
        moy = np.mean(scores)
        var = np.var(scores)
        stats.append({'Emotion': e, 'Moyenne': round(moy, 2), 'Variance': round(var, 2)})
    df_stats = pd.DataFrame(stats)
    st.dataframe(df_stats)
    bar_chart = alt.Chart(df_stats).mark_bar().encode(
         x=alt.X('Emotion:N', title='Émotion', scale=alt.Scale(scheme='category10')),
         y=alt.Y('Moyenne:Q', title='Moyenne'),
         tooltip=['Emotion', 'Moyenne']
    ).properties(title="Analyse globale des émotions : Moyennes", width=400, height=300)
    points = alt.Chart(df_stats).mark_point(filled=True, color='red', size=100).encode(
         x='Emotion:N',
         y='Variance:Q',
         tooltip=['Emotion', 'Variance']
    )
    global_chart = alt.layer(bar_chart, points).resolve_scale(y='independent')
    st.altair_chart(global_chart, use_container_width=True)

def analyser_globale(video_url, start_time, end_time, repertoire, extraction_interval, correction_synchro):
    st.write(f"Analyse globale de la vidéo de {start_time} s à {end_time} s")
    dossier_images = os.path.join(repertoire, "images_extraites")
    os.makedirs(dossier_images, exist_ok=True)
    video_path = telecharger_video(video_url, repertoire)
    fps_video = obtenir_fps(video_path)
    st.write(f"La vidéo est composée de {fps_video:.2f} FPS")
    detector = FER()
    results_images = []
    emotion_data_global = []
    images_data = []
    
    transcript_segments = obtenir_transcription_whisper(video_path, start_time, end_time, repertoire)
    
    for seconde in range(start_time, end_time + 1):
        imgs = extraire_images_custom_ffmpeg(video_path, dossier_images, seconde, extraction_interval)
        images_data.append(imgs)
        emotions_list = [analyser_image(img, detector) for img in imgs]
        nb_images = int(1 / extraction_interval)
        results_images.extend([{'Seconde': seconde, 'Frame': f'{nb_images}fps_{seconde * nb_images + idx}', **emo}
                               for idx, emo in enumerate(emotions_list)])
        moy_emotions, _ = emotion_dominante_par_moyenne(emotions_list)
        emotion_data_global.append({'Seconde': seconde, **moy_emotions})
    df_emotions = pd.DataFrame(results_images)
    df_emotion_global = pd.DataFrame(emotion_data_global)
    
    afficher_dataframe_et_streamgraph(df_emotions, df_emotion_global, start_time, end_time, repertoire)
    creer_concordancier(images_data, emotion_data_global, transcript_segments, repertoire, start_time, extraction_interval, [])
    stats_audio = analyser_audio_silence(video_path, start_time, end_time, repertoire)
    st.write(f"Temps de parole : {stats_audio['temps_parole']:.2f} s")
    st.write(f"Temps de silence : {stats_audio['temps_silence']:.2f} s")
    afficher_graph_temps_parole_silence(stats_audio)
    df_silence_detail = analyser_audio_silence_detail(video_path, start_time, end_time, extraction_interval, repertoire)
    st.subheader("Analyse détaillée des temps de silence par unité")
    st.dataframe(df_silence_detail)
    afficher_graph_silence_par_unite(df_silence_detail)
    
    # Export de la transcription pour vérification
    transcription_file = os.path.join(repertoire, "transcription.txt")
    with open(transcription_file, "w", encoding="utf-8") as f:
        for seg in transcript_segments:
            start_seg = float(seg['start'])
            end_seg = float(seg.get('end', start_seg + seg.get('duration', 0)))
            f.write(f"{start_seg:.2f} s - {end_seg:.2f} s : {seg['text']}\n")
    st.write(f"Transcription enregistrée dans : {transcription_file}")
    
    st.success("Analyse globale terminée.")

