import os
import struct
from collections import deque
from datetime import datetime

import google.generativeai as genai
import pvporcupine
import pyaudio
import webrtcvad
from dotenv import load_dotenv
from google.cloud import speech, texttospeech
from google.generativeai.types.generation_types import BlockedPromptException
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()

cloud_summit_information = """
# Vancouver Cloud Summit 2024 

## All the Clouds, All at once

**Date:** April 22nd 
**Venue:** Orpheum, 601 Smithe St, Vancouver, BC, V6B 3L4

Vancouver Cloud Summit 2024 is Western Canada's largest and most exciting cloud technology event. It offers a unique platform to learn, connect, and explore the endless possibilities of cloud computing.

## About

Whether you are a developer, IT professional, business leader, or simply curious about the cloud, CloudSummit offers a full day of expert speakers covering a wide range of topics on cloud technology, as well as a community hall for recruiters, startups, and local community groups to join. There will also be live demos, GenAI battles between Google, Microsoft and AWS, live racing of AI-powered cars, a live band playing all day, and much more!

## Speakers

### Community Speakers

**Mattias Andersson**
*Talk: Multi-Cloud and Serverless - Good or Bad?*
*Time: 10:50 am*
Mattias has talked at Microsoft Inspire, Microsoft Tech Skills Day, 100+ Training Videos for Pluralsight and A Cloud Guru, and much more.

**Payam Moghad**
*Talk: Develop Directly in the Cloud*
*Time: 12:30 pm*
Payam has talked recently at DevOps Days, Hashicorp, AWS Meetup and more.

**Amit Bajaj**
*Talk: Optimize your Cloud for Cost & Sustainability*
*Time: 1:50 pm*
Amit has talked at AWS Re:Invent, AWS OnAir Podcast and more.

### Guest Speakers

**Julia Furst Morgado**
*Talk: Head in the Clouds, Feet on the Ground: Practical Cloud Migration for Business Continuity*
*Time: 10:10 am*
Julia is a recognized leader in the tech community, serving as an AWS Community Builder, a CNCF Ambassador, a Google Women Techmakers Ambassador, and a Girl Code Ambassador. She actively supports and organizes the NY Code & Coffee Meetup and the KubeHuddle conference, fostering collaboration and learning opportunities.

**Tyler Mitchell**
*Talk: Future-Ready: Database Strategies for AI Innovation*
*Time: 11:30 am*
Tyler shares his insights from decades of experience in handling large datasets, exploring distributed data technology, and building solutions to extract value from disparate big data lakes. His role at Couchbase is to share the vision for how developers and architects can future-proof their work and take advantage of next-generation technology, today. Tyler is an O'Reilley author, and spent much of his time working in British Columbia for technology end natural resource companies. His early background is in geospatial technology and open source software advocacy.

**Chester Wisniewski**
*Talk: Cloud Security Strategies for 2024 and Beyond*
*Time: 1:10 pm*
Based in Vancouver, Chester regularly speaks at industry events, including RSA Conference, Virus Bulletin, Security BSides (Vancouver, London, Wales, Perth, Austin, Detroit, Los Angeles, Boston, and Calgary) and others. He's widely recognized as one of the industry's top security researchers and is regularly consulted by press, appearing on BBC News, ABC, NBC, Bloomberg, Washington Post, CBC, NPR, and more.

**Jeff Price**
*Talk: Leveling the Cloud Playing Field: Diversification Made Easy*
*Time: 2:30 pm*
Jeff is a Technologist, Strategist, Futurist, and Communicator.

## Schedule

**Main Stage**

- 9:30 am - Doors Open
- 10:00 am - Kickoff, welcome  
- 10:10 am - Julia Furst Morgado @ Veeam
- 10:40 am - Band & MC
- 10:50 am - Mattias Andersson  
- 11:20 am - Band & MC
- 11:30 am - Tyler Mitchell @ Couchbase
- 12:00 pm - LUNCH
- 12:30 pm - Payam Moghad
- 1:00 pm - Band & MC  
- 1:10 pm - Chester Wisniewski @ Sophos
- 1:40 pm - Band & MC
- 1:50 pm - Amit Bajaj
- 2:20 pm - Band & MC
- 2:30 pm - Jeff Price @ SUSE
- 3:00 pm - MC Wrap-up thankyou
- 3:10 pm - Band and Expo Hall
- 4:00 pm - Conclusion

**Explore Area (Open 10am - 4pm)**
- Sponsor Booths
- AWS/Azure/GCP AI Demo Labs  
- Local Technology Community Booth
- VIP Lounge

## Sponsors

- Digital Advertising Partner

All profits go to AWS User Group Meetup (AWSusergroups.com) and Azure User Group Meetup (AzureCanada.ca).

## Venue

**The Orpheum**
Situated in the heart of downtown Vancouver, the historic Orpheum is an iconic venue renowned for its stunning architecture and state-of-the-art facilities.

Full Address: 
Orpheum
601 Smithe St, Vancouver, BC, V6B 3L4
"""


def record_audio(audio_stream, sample_rate, frame_length, silent_seconds_required=1.5):
    print("Recording...")
    frames = []
    silent_seconds_count = 0.0
    consecutive_silent_seconds_required = silent_seconds_required
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # High aggressiveness
    frame_length = 320  # need to override this for webrtcvad

    try:
        # Record audio until silence is detected
        while silent_seconds_count < consecutive_silent_seconds_required:
            frame = audio_stream.read(frame_length, exception_on_overflow=False)
            frames.append(frame)
            is_speech = vad.is_speech(frame, sample_rate)

            if not is_speech:
                silent_seconds_count += frame_length / sample_rate
            else:
                # Reset count if speech is detected
                silent_seconds_count = 0
    except KeyboardInterrupt:
        print("Recording interrupted by user.")

    print("Recording stopped.")
    return b"".join(frames)


def transcribe_audio(client, audio_data):
    """Function to convert speech to text using Google Speech-to-Text."""
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)
    # Return text only if there are results
    if response.results:
        for result in response.results:
            print("Transcribed text: {}".format(result.alternatives[0].transcript))
        return response.results[0].alternatives[0].transcript
    else:
        print("No transcription results.")
        return None


def send_text_to_gemini(user_input, previous_texts, timestamp):
    # Combining past texts as context
    context = " ".join(previous_texts)

    # Adding system message
    system_message = f"""System Message - Your identity: Gemini, you are a smart, kind, and helpful AI assistant.
    You are at the Vancouver Cloud Summit and have the following information about the event: {cloud_summit_information}

    Never respond with code or markdown. Respond only with text as if it is a conversation.
    """

    # Initializing Gemini model
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    # Sending image and text instructions to the model
    prompt = f"{system_message}\nGiven the context: {context} and the current time: {timestamp}, please respond to the following message without repeating the context. Message: {user_input}"

    try:
        config = genai.GenerationConfig(max_output_tokens=1000)
        response = model.generate_content(
            [prompt], stream=True, generation_config=config
        )
        response.resolve()
        # Returning the generated text
        return response.text
    except BlockedPromptException:
        print(
            "AI response was blocked due to safety concerns. Please try a different input."
        )
        return "AI response was blocked due to safety concerns."


def text_to_speech_google(text, client):
    # Setting up the speech synthesis request
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Studio-O",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Sending the speech synthesis request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Saving the audio data to a file
    with open("./temp/output.mp3", "wb") as out:
        out.write(response.audio_content)

    # Loading the MP3 file
    sound = AudioSegment.from_mp3("./temp/output.mp3")
    # Adding a small amount of silence at the beginning
    silence = AudioSegment.silent(duration=300)
    sound_with_silence = silence + sound

    # Playing the sound
    play(sound_with_silence)


def main():
    # Loading the access key and keyword path from environment variables
    access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
    keyword_path = os.environ.get("PICOVOICE_KEYWORD_PATH")

    # Creating a Porcupine instance
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    # Initializing Google Cloud Speech-to-Text client
    speech_client = speech.SpeechClient()

    # Initializing PyAudio
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Initializing Google Cloud TTS API client
    tts_client = texttospeech.TextToSpeechClient()

    try:
        previous_texts = deque(maxlen=100)

        while True:
            try:
                # Reading audio data from PyAudio stream
                pcm = audio_stream.read(
                    porcupine.frame_length, exception_on_overflow=False
                )
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                # Detecting wake word using Porcupine
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word detected!")

                    # Continuing the process until silence is detected
                    while True:
                        audio_data = record_audio(
                            audio_stream, porcupine.sample_rate, porcupine.frame_length
                        )
                        user_input = transcribe_audio(speech_client, audio_data)

                        # Processing if there is voice input
                        if user_input:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Sending user input to Gemini AI model and generating a response
                            generated_text = send_text_to_gemini(
                                user_input, previous_texts, timestamp
                            )
                            print(
                                f"Timestamp: {timestamp}, Generated Text: {generated_text}"
                            )

                            # Updating past texts
                            previous_texts.append(
                                f"Timestamp: {timestamp}\nUser Message: {user_input}\nYour Response: {generated_text}\n"
                            )

                            # Converting AI response to speech and playing it
                            text_to_speech_google(generated_text, tts_client)

                        else:
                            print("Waiting for the keyword...")
                            break

            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("Input overflow, restarting the stream")
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                    if not audio_stream.is_stopped():
                        audio_stream.start_stream()
                else:
                    raise e
    except KeyboardInterrupt:
        print("\nExiting the program")
    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()


if __name__ == "__main__":
    main()
