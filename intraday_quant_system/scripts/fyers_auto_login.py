import os
import re
import json
import base64
import logging
import requests
import pyotp
import hmac
import hashlib
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv, set_key

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    fyersModel = None

logger = logging.getLogger(__name__)

def generate_fyers_token() -> str:
    """
    Headless Fyers API v3 Login.
    Requires FYERS_CLIENT_ID, FYERS_TOTP_SECRET, FYERS_PIN, FYERS_APP_ID, FYERS_SECRET_KEY in .env
    Returns the daily access token.
    """
    load_dotenv()
    
    fy_id = os.environ.get('FYERS_CLIENT_ID')
    totp_secret = os.environ.get('FYERS_TOTP_SECRET')
    pin = os.environ.get('FYERS_PIN')
    
    app_id = os.environ.get('FYERS_APP_ID') # e.g. ABCDEFGH-100
    secret_key = os.environ.get('FYERS_SECRET_KEY')
    redirect_uri = os.environ.get('FYERS_REDIRECT_URI', 'http://127.0.0.1:5000/login')
    
    if not all([fy_id, totp_secret, pin, app_id, secret_key]):
        logger.error("Missing required environment variables for headless login.")
        return None
        
    try:
        # Step 1: Request OTP/TOTP Flow
        res1 = requests.post(
            'https://api-t2.fyers.in/vagator/v2/send_login_otp_v2',
            json={"fy_id": base64.b64encode(f"{fy_id}".encode()).decode(), "app_id": "2"}
        )
        if res1.status_code != 200 or res1.json().get('s') != 'ok':
            logger.error(f"Failed at step 1: {res1.text}")
            return None
            
        request_key = res1.json().get('request_key')
        
        # Step 2: Verify TOTP
        totp = pyotp.TOTP(totp_secret).now()
        res2 = requests.post(
            'https://api-t2.fyers.in/vagator/v2/verify_totp',
            json={"request_key": request_key, "totp": totp}
        )
        if res2.status_code != 200 or res2.json().get('s') != 'ok':
            logger.error(f"Failed at step 2 (Verify TOTP): {res2.text}")
            return None
            
        request_key = res2.json().get('request_key')
        
        # Step 3: Verify PIN
        # Note: Depending on API version, identifier might need hashing. Typically plaintext is fine for v2 pin.
        res3 = requests.post(
            'https://api-t2.fyers.in/vagator/v2/verify_pin_v2',
            json={"request_key": request_key, "identity_type": "pin", "identifier": base64.b64encode(f"{pin}".encode()).decode()}
        )
        if res3.status_code != 200 or res3.json().get('s') != 'ok':
            logger.error(f"Failed at step 3 (Verify PIN): {res3.text}")
            return None
            
        access_token_vagator = res3.json()['data']['access_token']
        
        # Step 4: Generate Auth Code via APIv3
        app_id_hash = hashlib.sha256(f"{app_id}:{secret_key}".encode()).hexdigest()
        
        headers = {
            "Authorization": f"Bearer {access_token_vagator}",
            "Content-Type": "application/json"
        }
        
        # Fyers requires 'client_id' in payload which is actually the app_id without the -100 suffix sometimes, 
        # but standard API v3 uses app_id directly.
        payload = {
            "fyers_id": fy_id,
            "app_id": app_id[:-4] if app_id.endswith("-100") else app_id, 
            "redirect_uri": redirect_uri,
            "appType": "100",
            "code_challenge": "",
            "state": "None",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True
        }
        
        res4 = requests.post("https://api-t1.fyers.in/api/v3/token", headers=headers, json=payload)
        if res4.status_code != 200:
            logger.error(f"Failed at step 4 (Generate Auth Code): {res4.text}")
            return None
            
        auth_url = res4.json().get('Url')
        if not auth_url:
            logger.error(f"No auth URL returned: {res4.text}")
            return None
            
        parsed_url = urlparse(auth_url)
        auth_code = parse_qs(parsed_url.query).get('auth_code', [None])[0]
        
        if not auth_code:
            logger.error("Failed to extract auth_code from URL.")
            return None
            
        # Step 5: Convert Auth Code to Daily Access Token using the official SDK
        if not fyersModel:
            logger.error("fyers_apiv3 not found. Cannot complete flow.")
            return None
            
        session = fyersModel.SessionModel(
            client_id=app_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        
        session.set_token(auth_code)
        response = session.generate_token()
        
        if response.get('s') == 'ok':
            access_token = response['access_token']
            logger.info("Successfully generated daily Fyers Access Token!")
            
            # Save to .env
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            set_key(env_path, "FYERS_ACCESS_TOKEN", access_token)
            logger.info("Saved token to .env file.")
            return access_token
        else:
            logger.error(f"Failed to generate final token: {response}")
            return None
            
    except Exception as e:
        logger.exception(f"Exception during headless login: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    token = generate_fyers_token()
    if token:
        print("LOGIN SUCCESSFUL!")
    else:
        print("LOGIN FAILED. Check logs and .env file.")
