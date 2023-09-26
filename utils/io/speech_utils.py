import time
import azure.cognitiveservices.speech as speechsdk

def get_speech_input(prompt_text):
    print(prompt_text)
    words = recognize_from_microphone()
    sentence = []
    for word in words:
        print(f"{word['Word']}\t{word['Offset']}\t{word['Offset'] + word['Duration']}")
        sentence.append(word['Word'])
    return ' '.join(sentence)

SPEECH_KEY = '0d57bb33e99a412082d4dfd4d093e616'
SPEECH_REGION = 'westus'
def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language="en-US"
    speech_config.request_word_level_timestamps()
    speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "3000")

    audio_config = speechsdk.audio.AudioConfig(device_name="plughw:CARD=PCH,DEV=0")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    start = time.time()
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        stt = json.loads(speech_recognition_result.json)
        confidences_in_nbest = [item['Confidence'] for item in stt['NBest']]
        best_index = confidences_in_nbest.index(max(confidences_in_nbest))
        words = stt['NBest'][best_index]['Words']
        for word in words:
            word['Offset'] = start + word['Offset'] / 1e7
            word['Duration'] /= 1e7
        return words
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return []