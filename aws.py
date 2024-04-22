import contextlib
import json
import math
import os
import struct
import time
import wave
from collections import deque
from datetime import datetime
from threading import Thread

import boto3
import numpy as np
import pvporcupine
import pyaudio
import requests
import webrtcvad
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from cloud_summit import CLOUD_SUMMIT_INFORMATION

# Constants
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


def transcribe_audio(client, audio_data):
    """Function to convert speech to text using AWS Transcribe."""

    try:
        # Save audio data to a file
        audio_file_path = f"./temp/audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        with wave.open(audio_file_path, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            audio_file.setframerate(16000)
            audio_file.writeframes(audio_data)

        # Upload the audio file to S3
        s3_client = boto3.client("s3", region_name="us-west-2")
        bucket_name = "autimo-cloud-summit-audio"
        object_name = os.path.basename(audio_file_path)
        s3_client.upload_file(audio_file_path, bucket_name, object_name)

        # Start transcription job
        response = client.start_transcription_job(
            TranscriptionJobName=f"CloudSummit_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            Media={"MediaFileUri": f"s3://{bucket_name}/{object_name}"},
            MediaFormat="wav",
            LanguageCode="en-US",
            MediaSampleRateHertz=16000,
        )

        while True:
            # Get the latest status of the transcription job
            job_name = response["TranscriptionJob"]["TranscriptionJobName"]
            job_status_response = client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = job_status_response["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                result_url = job_status_response["TranscriptionJob"]["Transcript"][
                    "TranscriptFileUri"
                ]
                transcript_response = requests.get(result_url)
                transcript_data = transcript_response.json()
                return transcript_data["results"]["transcripts"][0]["transcript"]
            elif status == "FAILED":
                return None
            time.sleep(2)
    except Exception as e:
        console.log(f"Error transcribing audio: {e}")
        return None


def send_text_to_model(user_input, previous_texts, timestamp):
    # Combining past texts as context
    context = " ".join(previous_texts)

    # Adding system message
    system_message = f"""System Message - Your identity: Polly, you are a smart, kind, and helpful AI assistant.
    You are at the Vancouver Cloud Summit and have the following information about the event: {CLOUD_SUMMIT_INFORMATION}

    Never respond with code or markdown. Respond only with plain text as if it is transcript of a conversation.
    """

    # Initialize the Amazon Bedrock runtime client
    client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

    prompt = f"{system_message}\nGiven the context: {context} and the current time: {timestamp}, please respond to the following message without repeating the context. Message: {user_input}"

    try:
        response = client.invoke_model(
            modelId="mistral.mistral-large-2402-v1:0",
            body=json.dumps(
                {
                    "prompt": f"<s>[INST]{prompt}[/INST]",
                    "max_tokens": 1000,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 50,
                }
            ),
        )

        # Process and return the response
        result = json.loads(response.get("body").read())
        output_list = result.get("outputs", [])
        return " ".join([output["text"] for output in output_list])

    except Exception as err:
        console.log(f"Couldn't invoke Bedrock model. Here's why: {err}")
        return "There was an error with the model. Please try again."


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
                    title="Polly",
                )
            )
        start_time = end_time
        time.sleep(duration_per_word)


def text_to_speech(text, client, live_display):
    # Setting up the speech synthesis request
    output = "./temp/output.mp3"
    try:
        # Request speech synthesis using Polly
        response = client.synthesize_speech(
            Text=text, OutputFormat="mp3", VoiceId="Danielle", Engine="neural"
        )
    except Exception as error:
        # Handle possible errors gracefully
        console.log(f"Couldn't synthesize speech. Here's why: {error}")
        return

    if "AudioStream" in response:
        # Ensure the stream is properly closed after saving the audio
        with contextlib.closing(response["AudioStream"]) as stream:
            try:
                # Open a file for writing the output as a binary stream
                with open(output, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                # Could not write to file, exit gracefully
                console.log(error)
                return

    # Loading the MP3 file
    sound = AudioSegment.from_mp3(output)
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


def main():
    access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
    keyword_path = os.environ.get("PICOVOICE_KEYWORD_PATH")
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    speech_client = boto3.client("transcribe", region_name="us-west-2")

    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    tts_client = boto3.client("polly", region_name="us-west-2")

    previous_texts = deque(maxlen=100)

    console.clear()

    layout.split_column(
        Layout(name="current_state", ratio=1, minimum_size=6),
        Layout(name="ai_response", ratio=10),
    )
    layout["current_state"].update(Panel(Text("", style="bold"), title="Current State"))
    layout["ai_response"].update(Panel(Text(style="bold"), title="Polly"))

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
                            user_input = transcribe_audio(speech_client, audio_data)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            generated_text = send_text_to_model(
                                user_input, previous_texts, timestamp
                            )
                            previous_texts.append(
                                f"Timestamp: {timestamp}\nUser Message: {user_input}\nYour Response: {generated_text}\n"
                            )
                            text_to_speech(generated_text, tts_client, live)
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
