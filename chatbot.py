import azure.cognitiveservices.speech as speechsdk
import os
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from translate import Translator
import azure.cognitiveservices.speech as speechsdk

subscription_key = 'e8843e72c02f46f6b3e13afe06f92977'
region = 'eastus'


def call_chatbot(audio_file, emotion):
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_recognition_language = 'ar-EG'
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()
    # Process the recognition result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print('Recognized Text:', result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print('No speech could be recognized')
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print('Speech Recognition canceled:', cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print('Error details:', cancellation_details.error_details)
    arabic = result.text
    eng = translate_text_to_english(arabic)
    print("english text : " + eng)
    question = eng
    # emotion = "neutral"  # the variable to change with the detected emotion

    response = llm_chain.run(emotion=emotion, question=question)
    lines = [line for line in response.splitlines() if line.strip()]
    print("chatbot response : " + response)  # maybe changed with all lines

    ##trnalsate lines
    english_text = response
    arabic_text = translate_text_to_arabic(english_text)
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)

    # Set the output format and audio settings
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)

    # Set the voice and language
    speech_config.speech_synthesis_language = 'ar-EG'
    speech_config.speech_synthesis_voice_name = "ar-EG-SalmaNeural"
    # Set the input text
    text_to_speak = arabic_text
    # Set the output audio file path
    output_file = 'tarok1.wav'
    # Create a speech synthesizer object
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Use the synthesizer to generate the speech
    result = synthesizer.speak_text_async(text_to_speak).get()

    # Save the audio output to a file
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = result.audio_data
        with open(output_file, 'wb') as file:
            file.write(audio_data)
        print(f'Saved synthesized speech to {output_file}')
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f'Speech synthesis canceled. Error details: {cancellation_details.reason}')
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f'Error code: {cancellation_details.error_code}')
            print(f'Error details: {cancellation_details.error_details}')
    else:
        print('Speech synthesis failed')


os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# Run this if you have added HUGGINGFACEHUB_API_TOKEN as an environment variable
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

# huggingfacehub_api_token="your_api_token_here"


repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.6, "max_new_tokens": 500})

emotion_template = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
and also
You are an emotional supporter robot. The robot is here to help you with your daily tasks and provide emotional support.

{emotion} {question} 
"""

# prompt = PromptTemplate(template=template, input_variables=["question"])
prompt = PromptTemplate(template=emotion_template, input_variables=["emotion", "question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)


##arabic to english

# Function to translate English text to Arabic
def translate_text_to_english(text):
    translator = Translator(from_lang='ar', to_lang='en')
    translation = translator.translate(text)
    return translation


# Example usage
# Function to translate English text to Arabic
def translate_text_to_arabic(text):
    translator = Translator(to_lang='ar')
    translation = translator.translate(text)
    return translation


# call_chatbot('temp.wav', 'happy')
