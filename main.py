import os
import sys
import logging
import uvicorn
from typing import Optional
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pyngrok import ngrok

from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.telephony.server.base import TwilioInboundCallConfig, TelephonyServer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.logging import configure_pretty_logging
from memory_config import config_manager

# Configure Logging
configure_pretty_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
    auth_token=os.getenv("TWILIO_AUTH_TOKEN", "<your_twilio_auth_token>")
)
TWILIO_PHONE = os.getenv("OUTBOUND_CALLER_NUMBER", "<your_twilio_phone_number>")
logger.debug(f"Twilio configuration: SID={TWILIO_CONFIG.account_sid}, Phone={TWILIO_PHONE}")

# Config Manager
CONFIG_MANAGER = config_manager
logger.debug("Config manager initialized.")

# Agent Configuration
AGENT_CONFIG = ChatGPTAgentConfig(
    initial_message=BaseMessage(text="Hello! Welcome to Royal Apple Dental Clinic. How can I assist you today?"),
    prompt_preamble="You are a helpful assistant for a dental clinic. Assist customers with booking appointments, answering FAQs about services (like teeth cleaning, whitening, braces), and providing clinic timings.",
    generate_responses=True,
    openai_api_key=os.getenv("OPENAI_API_KEY", "<your_openai_api_key>")
)
logger.debug(f"Agent configuration: {AGENT_CONFIG}")

# Synthesizer Configuration
try:
    SYNTH_CONFIG = ElevenLabsSynthesizerConfig.from_telephone_output_device(
        api_key=os.getenv("ELEVEN_LABS_API_KEY", "<your-eleven-labs-api-key>")
    )
    logger.debug("Synthesizer initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing synthesizer: {e}")
    raise

# Telephony Server
telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=CONFIG_MANAGER,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=AGENT_CONFIG,
            twilio_config=TWILIO_CONFIG,
            synthesizer_config=SYNTH_CONFIG
        )
    ],
)
logger.debug("Telephony server initialized.")
app.include_router(telephony_server.get_router())

# Outbound Call Functionality
def start_outbound_call(to_phone: Optional[str]):
    if to_phone:
        try:
            outbound_call = OutboundCall(
                base_url=BASE_URL,
                to_phone=to_phone,
                from_phone=TWILIO_PHONE,
                config_manager=CONFIG_MANAGER,
                agent_config=AGENT_CONFIG,
                twilio_config=TWILIO_CONFIG,
                synthesizer_config=SYNTH_CONFIG
            )
            outbound_call.start()
            logger.debug(f"Outbound call started to {to_phone}")
        except Exception as e:
            logger.error(f"Error starting outbound call: {e}")

@app.get("/")
async def root(request: Request):
    env_vars = {
        "BASE_URL": BASE_URL,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ELEVEN_LABS_API_KEY": os.getenv("ELEVEN_LABS_API_KEY"),
        "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN": os.getenv("TWILIO_AUTH_TOKEN"),
        "OUTBOUND_CALLER_NUMBER": os.getenv("OUTBOUND_CALLER_NUMBER")
    }
    return templates.TemplateResponse("index.html", {"request": request, "env_vars": env_vars})

@app.post("/start_outbound_call")
async def api_start_outbound_call(to_phone: Optional[str] = Form(None)):
    start_outbound_call(to_phone)
    return {"status": "success"}

@app.post("/inbound_call")
async def inbound_call(request: Request):
    try:
        payload = await request.json()
        logger.debug(f"Inbound call payload: {payload}")
        return {"status": "call received"}
    except Exception as e:
        logger.error(f"Error processing inbound call: {e}")
        return {"status": "error"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
