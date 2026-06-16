import os
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

load_dotenv()

client_id = os.environ.get("FYERS_CLIENT_ID", "")
secret_key = os.environ.get("FYERS_SECRET_KEY", "")
redirect_uri = "http://127.0.0.1:5000/"  # Standard default URI

if not client_id or not secret_key:
    print("ERROR: FYERS_CLIENT_ID or FYERS_SECRET_KEY is missing from .env")
    exit(1)

session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type="code",
    grant_type="authorization_code"
)

response = session.generate_authcode()

print("=====================================================")
print("STEP 1: Click the URL below and log into Fyers:")
print("=====================================================\n")
print(response)
print("\n=====================================================")
print("STEP 2: After logging in, you will be redirected to a blank/error page.")
print("Look at the URL in your browser. Copy the string that comes after 'auth_code='")
print("=====================================================\n")

auth_code = input("Paste your auth_code here: ").strip()

if auth_code:
    session.set_token(auth_code)
    token_response = session.generate_token()
    
    if "access_token" in token_response:
        access_token = token_response["access_token"]
        print("\n✅ SUCCESS! Copy this token into your .env file as FYERS_ACCESS_TOKEN:\n")
        print(f"FYERS_ACCESS_TOKEN={access_token}\n")
    else:
        print("❌ Failed to generate access token:", token_response)