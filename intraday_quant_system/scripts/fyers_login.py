import os
import sys
import webbrowser
from urllib.parse import urlparse, parse_qs
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

# Load env variables
dotenv_path = os.path.expanduser("~/.env")
load_dotenv(dotenv_path)

APP_ID = os.getenv("FYERS_APP_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_ID")
REDIRECT_URI = "http://127.0.0.1:5000/"

if not APP_ID or not SECRET_KEY:
    print("ERROR: FYERS_APP_ID or FYERS_SECRET_ID not found in ~/.env")
    sys.exit(1)

def main():
    print("--- Fyers API OAuth Login ---")
    session = fyersModel.SessionModel(
        client_id=APP_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )

    # 1. Generate Auth URL
    auth_link = session.generate_authcode()
    print(f"\n[ACTION REQUIRED] Please open the following URL to login:")
    print(auth_link)
    print("\nIf you are on a desktop, it may open automatically...")
    
    try:
        webbrowser.open(auth_link)
    except:
        pass

    # 2. User pastes the redirected URL (with the auth_code)
    print("\nAfter logging in, you will be redirected to a page that says 'Success' or similar.")
    print("Copy the ENTIRE URL of that redirected page and paste it below.")
    redirected_url = input("\nEnter the full redirected URL: ").strip()

    # 3. Extract the auth_code
    parsed_url = urlparse(redirected_url)
    auth_code = parse_qs(parsed_url.query).get('auth_code')

    if not auth_code:
        print("ERROR: Could not find 'auth_code' in the provided URL.")
        sys.exit(1)
        
    auth_code = auth_code[0]
    print(f"Auth Code Extracted: {auth_code}")

    # 4. Exchange Auth Code for Access Token
    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("s") != "ok":
        print(f"ERROR: Failed to generate token: {response}")
        sys.exit(1)

    access_token = response["access_token"]
    
    # 5. Save securely to ~/.env
    print("\nSuccessfully generated Access Token!")
    
    # Append to .env
    with open(dotenv_path, "a") as f:
        f.write(f"\nFYERS_ACCESS_TOKEN={access_token}\n")
        
    print(f"Access Token saved securely to {dotenv_path}")
    print("\nYou are now ready for Paper Trading / Live Execution!")

if __name__ == "__main__":
    main()
