import gradio as gr
import whisper
from transformers import pipeline
import requests
import cv2
import string
import numpy as np
import tensorflow as tf
import edge_tts
import asyncio
import tempfile

# Load models
whisper_model = whisper.load_model("base")
sentiment_analysis = pipeline(
    "sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")


def load_sign_language_model():
    return tf.keras.models.load_model('best_model.h5')


sign_language_model = load_sign_language_model()

# Get all available voices


async def get_voices():
    voices = await edge_tts.list_voices()
    return {f"{v['ShortName']} - {v['Locale']} ({v['Gender']})": v['ShortName'] for v in voices}

# Audio-based functions


def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score']
                         for result in results}
    return sentiment_results


def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment}: {score}\n"
    return sentiment_text


def search_text(text, api_key):
    api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": text}]}]}

    try:
        response = requests.post(
            api_endpoint, headers=headers, json=payload, params={"key": api_key})
        response.raise_for_status()
        response_json = response.json()
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            content_parts = response_json['candidates'][0]['content']['parts']
            if len(content_parts) > 0:
                return content_parts[0]['text'].strip()
        return "No relevant content found."
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


async def text_to_speech(text, voice, rate, pitch):
    if not text.strip():
        return None, gr.Warning("Please enter text to convert.")
    if not voice:
        return None, gr.Warning("Please select a voice.")

    voice_short_name = voice.split(" - ")[0]
    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}Hz"
    communicate = edge_tts.Communicate(
        text, voice_short_name, rate=rate_str, pitch=pitch_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    return tmp_path, None


async def tts_interface(text, voice, rate, pitch):
    audio, warning = await text_to_speech(text, voice, rate, pitch)
    return audio, warning


def inference_audio(audio, sentiment_option, api_key, tts_voice, tts_rate, tts_pitch):
    if audio is None:
        return "No audio file provided.", "", "", "", None

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    _, probs = whisper_model.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)

    sentiment_results = analyze_sentiment(result.text)
    sentiment_output = display_sentiment_results(
        sentiment_results, sentiment_option)

    search_results = search_text(result.text, api_key)

    # Generate audio for explanation
    explanation_audio, _ = asyncio.run(tts_interface(
        search_results, tts_voice, tts_rate, tts_pitch))

    return lang.upper(), result.text, sentiment_output, search_results, explanation_audio

# Image-based functions


def get_explanation(letter, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": f"Explain how the American Sign Language letter '{letter}' is shown, its significance, and why it is represented this way."}]}
        ]
    }
    params = {"key": api_key}

    try:
        response = requests.post(url, headers=headers,
                                 json=data, params=params)
        response.raise_for_status()
        response_data = response.json()
        explanation = response_data.get("contents", [{}])[0].get("parts", [{}])[
            0].get("text", "No explanation available.")
        # Remove unnecessary symbols and formatting
        explanation = explanation.replace(
            "*", "").replace("#", "").replace("$", "").replace("\n", " ").strip()
        # Remove additional special characters, if needed
        explanation = explanation.translate(
            str.maketrans('', '', string.punctuation))
        return explanation
    except requests.RequestException as e:
        return f"Error fetching explanation: {e}"


def classify_sign_language(image, api_key):
    img = np.array(image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (28, 28))
    normalized_img = gray_img / 255.0
    input_img = np.expand_dims(normalized_img, axis=0)

    output = sign_language_model.predict(input_img)
    output = np.argmax(output, axis=1).item()
    uppercase_alphabet = string.ascii_uppercase
    output = output + 1 if output > 7 else output
    pred = uppercase_alphabet[output]

    explanation = get_explanation(pred, api_key)

    return pred, explanation

# Gradio interface


def process_input(input_type, audio=None, image=None, sentiment_option=None, api_key=None, tts_voice=None, tts_rate=0, tts_pitch=0):
    if input_type == "Audio":
        return inference_audio(audio, sentiment_option, api_key, tts_voice, tts_rate, tts_pitch)
    elif input_type == "Image":
        pred, explanation = classify_sign_language(image, api_key)
        explanation_audio, _ = asyncio.run(tts_interface(
            explanation, tts_voice, tts_rate, tts_pitch))
        return "N/A", pred, "N/A", explanation, explanation_audio


async def main():
    voices = await get_voices()

    with gr.Blocks() as demo:
        gr.Markdown("# Speak & Sign AI Assistant")

        # Layout: Split user input and bot response sides
        with gr.Row():
            # User Input Side
            with gr.Column():
                gr.Markdown("### User Input")
                # Input selection
                input_type = gr.Radio(label="Choose Input Type", choices=[
                                      "Audio", "Image"], value="Audio")

                # API key input
                api_key_input = gr.Textbox(
                    label="API Key", placeholder="Your API key here", type="password")

                # Audio input
                audio_input = gr.Audio(
                    label="Upload or Record Audio", type="filepath", visible=True)
                sentiment_option = gr.Radio(choices=[
                                            "Sentiment Only", "Sentiment + Score"], label="Sentiment Output", value="Sentiment Only", visible=True)

                # Image input
                image_input = gr.Image(
                    label="Upload Image", type="pil", visible=False)

                # TTS settings for explanation
                tts_voice = gr.Dropdown(label="Select Voice", choices=[
                ] + list(voices.keys()), value="")
                tts_rate = gr.Slider(
                    minimum=-50, maximum=50, value=0, label="Speech Rate Adjustment (%)", step=1)
                tts_pitch = gr.Slider(
                    minimum=-20, maximum=20, value=0, label="Pitch Adjustment (Hz)", step=1)

                # Change input visibility based on selection
                def update_visibility(input_type):
                    if input_type == "Audio":
                        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

                input_type.change(update_visibility, inputs=input_type, outputs=[
                                  audio_input, sentiment_option, image_input])

                # Submit button
                submit_btn = gr.Button("Submit")

            # Bot Response Side
            with gr.Column():
                gr.Markdown("### Bot Response")

                lang_str = gr.Textbox(
                    label="Detected Language", interactive=False)
                text = gr.Textbox(
                    label="Transcription or Prediction", interactive=False)
                sentiment_output = gr.Textbox(
                    label="Sentiment Analysis Results", interactive=False)
                search_results = gr.Textbox(
                    label="Explanation or Search Results", interactive=False)
                audio_output = gr.Audio(
                    label="Generated Explanation Audio", type="filepath", interactive=False)

        # Submit button action
        submit_btn.click(
            process_input,
            inputs=[input_type, audio_input, image_input, sentiment_option,
                    api_key_input, tts_voice, tts_rate, tts_pitch],
            outputs=[lang_str, text, sentiment_output,
                     search_results, audio_output]
        )

    demo.launch(share=True)

asyncio.run(main())
