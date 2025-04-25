#Demo2 AZURE synth configured
#importing various libs / packager
import os
import sys
import logging
import asyncio
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pyngrok import ngrok


import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.server.base import TwilioInboundCallConfig, TelephonyServer
from memory_config import config_manager

# Configure Logging
configure_pretty_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI(docs_url=None)
templates = Jinja2Templates(directory="templates")

# Ngrok setup for local development
BASE_URL = os.getenv("BASE_URL")
if not BASE_URL:
    ngrok_auth = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_auth:
        ngrok.set_auth_token(ngrok_auth)
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 3000
    BASE_URL = ngrok.connect(port).public_url.replace("https://", "")
logger.debug(f"BASE_URL set to: {BASE_URL}")

# Twilio Configuration
TWILIO_CONFIG = TwilioConfig(
    account_sid=os.getenv("TWILIO_ACCOUNT_SID", "<your_twilio_account_sid>"),
    auth_token=os.getenv("TWILIO_AUTH_TOKEN", "<your_twilio_auth_token>"),
)
TWILIO_PHONE = os.getenv("OUTBOUND_CALLER_NUMBER", "<your_twilio_phone_number>")
logger.debug(f"Twilio configuration: SID={TWILIO_CONFIG.account_sid}, Phone={TWILIO_PHONE}")

# Config Manager
CONFIG_MANAGER = config_manager
logger.debug("Config manager initialized.")

# AWS Polly and Transcribe Clients
polly_client = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)

transcribe_client = TranscribeStreamingClient(
    region=os.getenv("AWS_REGION", "us-east-1"),
)

# AWS Polly TTS Function
def synthesize_speech(text: str, voice_id: str = "Joanna", output_format: str = "mp3") -> bytes:
    """
    Use AWS Polly to generate speech from text.
    """
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            VoiceId=voice_id,
            OutputFormat=output_format,
        )
        audio_stream = response["AudioStream"].read()
        logger.debug("Polly synthesis successful")
        return audio_stream
    except Exception as e:
        logger.error(f"Error in Polly TTS: {e}")
        raise

# Custom handler to process real-time transcription results
class MyTranscriptResultHandler(TranscriptResultStreamHandler):
    def __init__(self, result_stream):
        super().__init__(result_stream)
        self.transcription = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if result.is_partial is False:
                self.transcription += " ".join([alt.transcript for alt in result.alternatives])
                logger.debug(f"Transcription so far: {self.transcription}")

# Real-time transcription function
async def transcribe_audio_stream(audio_stream: AsyncGenerator[bytes, None]) -> str:
    """
    Transcribe real-time audio using AWS Transcribe Streaming.
    """
    try:
        result_stream = await transcribe_client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=8000,
            media_encoding="pcm",
            audio_stream=audio_stream,
        )
        handler = MyTranscriptResultHandler(result_stream.transcript_result_stream)
        await handler.handle_events()
        return handler.transcription
    except Exception as e:
        logger.error(f"Error in Transcribe streaming: {e}")
        raise

# Agent Configuration
AGENT_CONFIG = ChatGPTAgentConfig(
    initial_message=BaseMessage(text="Hello! Welcome to Royal Apple Dental Clinic. How can I assist you today?"),
    prompt_preamble="You are a helpful assistant for a dental clinic. Assist customers with booking appointments, answering FAQs about services (like teeth cleaning, whitening, braces), and providing clinic timings.",
    generate_responses=True,
    openai_api_key=os.getenv("OPENAI_API_KEY", "<your_openai_api_key>"),
)
logger.debug(f"Agent configuration: {AGENT_CONFIG}")

# Custom Transcriber and Synthesizer Setup for Vocode
class CustomTranscriber:
    async def transcribe(self, audio_stream: AsyncGenerator[bytes, None]) -> str:
        return await transcribe_audio_stream(audio_stream)

class CustomSynthesizer:
    def synthesize(self, text: str) -> bytes:
        return synthesize_speech(text)

# Telephony Server
telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=CONFIG_MANAGER,  # Replace with actual config manager if needed
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=AGENT_CONFIG,
            twilio_config=TWILIO_CONFIG,
            synthesizer=CustomSynthesizer(),
            transcriber=CustomTranscriber(),
        )
    ],
)
logger.debug("Telephony server initialized.")
app.include_router(telephony_server.get_router())

# Root Endpoint
@app.get("/")
async def root(request: Request):
    env_vars = {
        "BASE_URL": BASE_URL,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": os.getenv("AWS_REGION"),
        "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN": os.getenv("TWILIO_AUTH_TOKEN"),
        "OUTBOUND_CALLER_NUMBER": os.getenv("OUTBOUND_CALLER_NUMBER"),
    }
    return templates.TemplateResponse("index.html", {"request": request, "env_vars": env_vars})

# Run the Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
