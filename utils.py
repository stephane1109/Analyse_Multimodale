# utils.py
import os
import subprocess
import re
import math
import shutil
import numpy as np
import pandas as pd
import cv2
from yt_dlp import YoutubeDL
import whisper
from fer import FER
from youtube_transcript_api import YouTubeTranscriptApi

def telecharger_video(url, repertoire):
    """
    Télécharge la vidéo depuis l'URL et la sauvegarde dans le répertoire spécifié.
    """
    video_path = os.path.join(repertoire, 'video.mp4')
    if os.path.exists(video_path):
        return video_path
    ydl_opts = {'outtmpl': video_path, 'format': 'best'}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return video_path

def obtenir_fps(video_path):
    """
    Renvoie le nombre de FPS de la vidéo.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def extraire_images_custom_ffmpeg(video_path, repertoire, seconde, extraction_interval):
    """
    Extrait des images pour une seconde donnée selon extraction_interval.
    Pour 25 fps, extraction_interval doit être fixé à 0.04 s.
    """
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
            # Vous pouvez ajouter un log ici si nécessaire
            break
        images_extraites.append(image_path)
    return images_extraites

def analyser_image(image_path, detector):
    """
    Analyse l'image avec FER, dessine un rectangle autour du visage et affiche les scores arrondis à 2 décimales.
    """
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
    """
    Pour une liste de dictionnaires d'émotions (une par image),
    calcule la moyenne pour chaque émotion et renvoie l'émotion dominante.
    """
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    moyenne_emotions = {e: round(np.mean([emo.get(e, 0) for emo in emotions_list]), 2) for e in emotions}
    emotion_dominante = max(moyenne_emotions, key=moyenne_emotions.get)
    return moyenne_emotions, emotion_dominante

def obtenir_transcription_whisper(video_path, start_time, end_time, repertoire):
    """
    Extrait l'audio de la portion [start_time, end_time] de la vidéo et utilise OpenAI Whisper
    pour obtenir une transcription. Renvoie la liste des segments (chaque segment contient 'start', 'duration' et 'text').
    """
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
    """
    Pour un timestamp (float), renvoie le texte du segment couvrant ce timestamp en appliquant une correction.
    Si aucun segment ne correspond exactement, retourne le texte du segment le plus proche.
    """
    ts = timestamp + correction
    for segment in transcript_segments:
        start = float(segment['start'])
        duration = float(segment.get('duration', 0))
        if start <= ts < start + duration:
            return segment['text']
    closest_segment = None
    min_diff = float('inf')
    for segment in transcript_segments:
        diff = abs(ts - float(segment['start']))
        if diff < min_diff:
            min_diff = diff
            closest_segment = segment
    return closest_segment['text'] if closest_segment else ""

def detecter_pauses(video_path, start_time, end_time, repertoire):
    """
    Détecte les pauses (silence > 1 s) dans l'audio extrait de la vidéo.
    Renvoie une liste de dictionnaires contenant 'start', 'duration' et 'end' pour chaque pause.
    """
    audio_path = os.path.join(repertoire, "audio_segment.wav")
    cmd_extract = ['ffmpeg', '-y', '-i', video_path, '-ss', str(start_time), '-to', str(end_time),
                   '-vn', '-acodec', 'pcm_s16le', audio_path]
    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_silence = ['ffmpeg', '-i', audio_path, '-af', 'silencedetect=noise=-30dB:d=0.5', '-f', 'null', '-']
    result = subprocess.run(cmd_silence, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stderr.decode('utf-8')
    pauses = []
    matches = re.findall(r"silence_start: ([0-9.]+).*?silence_duration: ([0-9.]+)", output, re.DOTALL)
    for s, d in matches:
        s, d = float(s), float(d)
        if d > 1.0 and start_time <= s < end_time:
            pauses.append({"start": s, "duration": d, "end": s + d})
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return pauses

def exporter_concordancier_pause(pause, images_data, transcript_segments, start_time, end_time, base_export_directory):
    """
    Pour un contexte de pause, exporte un concordancier contenant les timestamps, le nom d'image et la transcription.
    """
    window_start = max(start_time, pause["start"] - 2)
    window_end = min(end_time, pause["end"] + 2)
    pause_dir = os.path.join(base_export_directory, f"pause_{window_start:.2f}_{window_end:.2f}")
    os.makedirs(pause_dir, exist_ok=True)
    data = {"Timestamp": [], "Image": [], "Transcription": []}
    t = window_start
    while t < window_end:
        texte = get_transcript_text_at(t, transcript_segments)
        sec_index = int(math.floor(t)) - start_time
        img_file = os.path.basename(images_data[sec_index][0]) if sec_index < len(images_data) and images_data[sec_index] else ""
        data["Timestamp"].append(t)
        data["Image"].append(img_file)
        data["Transcription"].append(texte)
        t += 0.04
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(pause_dir, "concordance_pause.xlsx"), index=False)
    df.to_csv(os.path.join(pause_dir, "concordance_pause.csv"), index=False)
    return pause_dir, df

def analyser_audio_silence(video_path, start_time, end_time, repertoire):
    """
    Analyse globale de l'audio entre start_time et end_time pour calculer les temps de parole et de silence.
    """
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
    """
    Analyse détaillée de l'audio pour calculer le temps de silence par unité (selon extraction_interval).
    Renvoie un DataFrame avec les timestamps et le temps de silence correspondant.
    """
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
    for s, d in matches:
        s, d = float(s), float(d)
        if start_time <= s < end_time:
            bin_index = int(math.floor((s - start_time) / extraction_interval))
            bin_key = start_time + bin_index * extraction_interval
            bins[bin_key] += d
    df = pd.DataFrame({
        'Temps': list(bins.keys()),
        'Temps_Silence': [round(val, 2) for val in bins.values()]
    })
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return df
