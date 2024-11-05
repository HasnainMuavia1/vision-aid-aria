from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import requests
from openai import OpenAI
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os
from dotenv import load_dotenv
from django.conf import settings
import PyPDF2
import docx
load_dotenv()

# OpenAI and IBM Watson setup
base_url = os.environ.get('OPENAI_BASE_URL')
api_key = os.environ.get('ARIA_API_KEY')
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# IBM Text to Speech and Speech to Text configuration
ibm_key = os.environ.get('IBM_KEY_TEXT_TO_SPEECH')
# print("key", ibm_key)
ibm_speech_to_text = os.environ.get('IBM_KEY_SPEECH_TO_TEXT')
ibm_tts_url = os.environ.get('IBM_TTS_URL')
ibm_stt_url = os.environ.get('IBM_STT_URL')

# Set up IBM Watson services
authenticator = IAMAuthenticator(ibm_speech_to_text)
speech_to_text_service = SpeechToTextV1(authenticator=authenticator)
speech_to_text_service.set_service_url(ibm_stt_url)

# Global variable for storing image data
global_base64_image = None


@csrf_exempt
def index(request):
    global global_base64_image

    if request.method == "POST":
        # Check if an audio file is uploaded
        if 'audio' in request.FILES:
            audio_file = request.FILES['audio']
            print("audio received", audio_file)
            audio_data = audio_file.read()

            # Convert speech to text using IBM Watson Speech to Text
            transcribed_text = speech_to_text(audio_data)
            print("transcribed", transcribed_text)

            # Generate response using image (if exists) and transcribed text
            response_text = generate_prompt(global_base64_image, transcribed_text)
            print(response_text)

            # Convert the generated response to audio using IBM Text to Speech
            audio_content = text_to_speech(response_text)

            return JsonResponse({'status': 'success', 'audio': audio_content, 'text': response_text})

        # Check if an image file is uploaded (without audio)
        elif 'file_img' in request.FILES:
            image = request.FILES['file_img']
            image_data = image.read()
            global_base64_image = base64.b64encode(image_data).decode('utf-8')
            default_prompt = "As a medical assistant, describe the disease or fracture depicted in the image in detail, without suggesting any treatment or management steps."

            # Generate response using only the image and default prompt
            response_text = generate_prompt(global_base64_image, default_prompt)

            # Convert the response to audio using IBM Text to Speech
            audio_content = text_to_speech(response_text)

            return JsonResponse({'status': 'success', 'audio': audio_content, 'text': response_text})

        # Check if a document file is uploaded (PDF, DOCX, or text)
        elif 'file_img' in request.FILES:  # Keep the same file name as per your instruction
            file = request.FILES['file_img']
            file_extension = file.name.split('.')[-1].lower()

            # Extract text based on file type
            if file_extension == 'pdf':
                extracted_text = extract_text_from_pdf(file)
            elif file_extension in ['doc', 'docx']:
                extracted_text = extract_text_from_docx(file)
            elif file_extension == 'txt':
                extracted_text = file.read().decode('utf-8')
            else:
                return JsonResponse({'status': 'error', 'message': 'Unsupported file format.'})

            # Generate response using the extracted text
            response_text = generate_prompt_text(extracted_text)

            # Convert the response to audio using IBM Text to Speech
            audio_content = text_to_speech(response_text)

            return JsonResponse({'status': 'success', 'audio': audio_content, 'text': response_text})

        # If no files are uploaded
        return JsonResponse({'status': 'error', 'message': 'No file uploaded.'})

    # On GET request, don't clear the image
    return render(request, 'chat.html')

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()
    return text

def generate_prompt(image, text):
    response = client.chat.completions.create(
        model="aria",
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant for blind and visually impaired people. You will be given an "
                           "image, and your responses should provide clear, accessible insights related to the user's "
                           "needs, such as detecting fractures or diseases from images like MRIs or X-rays. Since "
                           "your responses will be transcribed to voice, please answer in simple, plain text."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"<image>\n{text}"
                    }
                ]
            }

        ],
        stream=False,
        temperature=0.6,
        max_tokens=1024,
        top_p=1,
        stop=["<|im_end|>"]
    )

    return response.choices[0].message.content


def text_to_speech(text):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {base64.b64encode(f"apikey:{ibm_key}".encode()).decode()}'
    }

    data = {
        "text": text,
        "voice": "en-US_AllisonV3Voice",
        "accept": "audio/wav"
    }

    response = requests.post(f"{ibm_tts_url}/v1/synthesize", headers=headers, json=data)

    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        print("Error in Text to Speech API:", response.status_code, response.text)
        return None


def speech_to_text(audio_data):
    try:
        response = speech_to_text_service.recognize(
            audio=audio_data,
            content_type='audio/wav',
            model='en-US_BroadbandModel'
        ).get_result()

        if response['results']:
            transcript = response['results'][0]['alternatives'][0]['transcript']
            return transcript.strip()
        else:
            return "No transcription available."

    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error occurred during transcription."


def generate_prompt_text(text):
    response = client.chat.completions.create(
        model="aria",
        messages=[
            {"role": "system",
             "content": "You are a medical assistant for blind and visually impaired people. You will be given text "
                        "related to prescriptions or other medical information, and your responses should be clear, "
                        "accessible, and relevant to the user's needs. as your repsonse will be transcribed to voice "
                        "so your answer must be plain texts"},
            {"role": "user", "content": f"{text}"}
        ],
        stop=["<|im_end|>"],
        stream=False,
        temperature=0.6,
        max_tokens=1024,
        top_p=1
    )

    # Check if there is a valid response and print the content

    return response.choices[0].message.content
