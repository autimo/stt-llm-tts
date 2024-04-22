import math
import os
import struct
import time
import wave
from collections import deque
from datetime import datetime
from threading import Thread

import azure.cognitiveservices.speech as azure_speech
import numpy as np
import pvporcupine
import pyaudio
import webrtcvad
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydub import AudioSegment
from pydub.playback import play
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from cloud_summit import CLOUD_SUMMIT_INFORMATION

SCALE_MIN = 1
SCALE_MAX = 10
RMS_MAX_EXPECTED = 200

load_dotenv()

# Initialize rich console
console = Console(record=True)
layout = Layout(name="root")


def record_audio(
    audio_stream,
    sample_rate,
    frame_length,
    silent_seconds_required=1.5,
    live_display=None,
):
    frames = []
    silent_seconds_count = 0.0
    consecutive_silent_seconds_required = silent_seconds_required
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # High aggressiveness
    frame_length = 320  # need to override this for webrtcvad

    try:
        while silent_seconds_count < consecutive_silent_seconds_required:
            frame = audio_stream.read(frame_length, exception_on_overflow=False)
            frames.append(frame)
            is_speech = vad.is_speech(frame, sample_rate)

            if not is_speech:
                silent_seconds_count += frame_length / sample_rate
            else:
                silent_seconds_count = 0

            if live_display:
                audio_data = np.frombuffer(frame, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data**2))
                if rms == 0:
                    scaled_volume = 0
                else:
                    volume = 10 * math.log10(rms)
                    scaled_volume = int(
                        max(
                            SCALE_MIN,
                            min(SCALE_MAX, volume / math.log10(RMS_MAX_EXPECTED)),
                        )
                    )
                layout["current_state"].update(
                    Panel(
                        Text(
                            "\nListening...\n\n" + "🔊" * scaled_volume,
                            style="bold green" if is_speech else "dim",
                            justify="center",
                        ),
                        title="Current State",
                    )
                )

    except KeyboardInterrupt:
        console.log("Recording interrupted by user.")

    return b"".join(frames)


def transcribe_audio(audio_data):
    """Function to convert speech to text using Azure Speech-to-Text."""

    try:
        # Save audio data to a file
        audio_file_path = "./temp/audio.wav"
        with wave.open(audio_file_path, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            audio_file.setframerate(16000)
            audio_file.writeframes(audio_data)
        speech_config = azure_speech.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        audio_input = azure_speech.AudioConfig(filename=audio_file_path)

        speech_recognizer = azure_speech.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_input
        )

        result = speech_recognizer.recognize_once()
        if result.reason == azure_speech.ResultReason.RecognizedSpeech:
            return result.text
        else:
            return None
    except Exception as e:
        console.log(f"Failed to transcribe audio: {str(e)}")
        return None


def send_text_to_model(client, user_input, previous_texts, timestamp):
    # Combining past texts as context
    context = " ".join(previous_texts)

    # Adding system message
    system_message = f"""System Message - Your identity: Ava, you are a smart, kind, and helpful AI assistant.
    You are at the Vancouver Cloud Summit and have the following information about the event: {CLOUD_SUMMIT_INFORMATION}

    Never respond with code or markdown. Respond only with plain text as if it is transcript of a conversation.
    """

    prompt = f"{system_message}\nGiven the context: {context} and the current time: {timestamp}, please respond to the following message without repeating the context. Message: {user_input}"

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        console.log(f"Failed to generate response from Azure OpenAI: {str(e)}")
        return None


def update_speech_display(text, duration, live_display):
    words = text.split()
    duration_per_word = duration / len(words) if words else 0
    start_time = 0
    for i, word in enumerate(words):
        end_time = start_time + duration_per_word

        if live_display:
            num_dots = i % 3 + 1
            layout["current_state"].update(
                Panel(
                    Text(
                        f"\n{'.' * (num_dots)}Responding{'.' * num_dots}",
                        style="bold green",
                        justify="center",
                    ),
                    title="Current State",
                )
            )
            layout["ai_response"].update(
                Panel(
                    Text(f"\n{' '.join(words[:i+1])}", style="bold"),
                    title="Ava",
                )
            )
        start_time = end_time
        time.sleep(duration_per_word)


def text_to_speech(text, live_display):
    """Function to convert text to speech using Azure Text-to-Speech."""

    try:
        if text is None:
            return

        speech_config = azure_speech.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"

        wav_filename = "./temp/output.wav"
        audio_config = azure_speech.AudioConfig(filename=wav_filename)

        synthesizer = azure_speech.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )

        console.log("Speaking...: ", text)
        result = synthesizer.speak_text(text)
        console.log("Speaking done: ", result)

        # Loading the MP3 file
        sound = AudioSegment.from_wav(wav_filename)
        # Adding a small amount of silence at the beginning
        silence = AudioSegment.silent(duration=300)
        sound_with_silence = silence + sound

        # Playing the sound and updating the live display with the waveform one word at a time
        duration = sound_with_silence.duration_seconds

        # Run playing sound and updating display in parallel
        play_thread = Thread(target=play, args=(sound_with_silence,))
        display_thread = Thread(
            target=update_speech_display, args=(text, duration, live_display)
        )

        play_thread.start()
        display_thread.start()

        play_thread.join()
        display_thread.join()
    except Exception as e:
        console.log(f"Failed to generate speech: {str(e)}")


def main():
    access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
    keyword_path = os.environ.get("PICOVOICE_KEYWORD_PATH")
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    # openai_client = openai.OpenAI(
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     base_url=os.getenv("AZURE_OPENAI_BASE"),
    # )
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_BASE"),
    )

    previous_texts = deque(maxlen=10)

    console.clear()

    layout.split_column(
        Layout(name="current_state", ratio=1, minimum_size=6),
        Layout(name="ai_response", ratio=10),
    )
    layout["current_state"].update(Panel(Text("", style="bold"), title="Current State"))
    layout["ai_response"].update(Panel(Text(style="bold"), title="Ava"))

    with Live(renderable=layout, console=console, refresh_per_second=10) as live:
        try:
            while True:
                pcm = audio_stream.read(
                    porcupine.frame_length, exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm)
                layout["current_state"].update(
                    Panel(
                        Text(
                            "\n👂 Waiting for the keyword 👂",
                            style="bold",
                            justify="center",
                        ),
                        title="Current State",
                    )
                )

                if keyword_index >= 0:
                    console.log("Wake word detected!")
                    empty_inputs = 0
                    while empty_inputs < 3:
                        audio_data = record_audio(
                            audio_stream,
                            porcupine.sample_rate,
                            porcupine.frame_length,
                            live_display=live,
                        )
                        if len(audio_data) > 48000:
                            user_input = transcribe_audio(audio_data)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            generated_text = send_text_to_model(
                                openai_client, user_input, previous_texts, timestamp
                            )
                            previous_texts.append(
                                f"Timestamp: {timestamp}\nUser Message: {user_input}\nYour Response: {generated_text}\n"
                            )
                            text_to_speech(generated_text, live)
                            empty_inputs = 0
                        else:
                            empty_inputs += 1
                            console.log(
                                f"Empty input detected. {3 - empty_inputs} attempts remaining."
                            )
                    if empty_inputs == 3:
                        console.log(
                            "No valid input after 3 attempts. Waiting for the keyword..."
                        )
        except KeyboardInterrupt:
            console.log("\nExiting the program")
        finally:
            audio_stream.close()
            pa.terminate()
            porcupine.delete()


if __name__ == "__main__":
    main()
