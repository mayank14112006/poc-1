import os
from infisical_sdk import InfisicalSDKClient

client_id = os.getenv("INFISICAL_CLIENT_ID", "your_infisical_client_id")
client_secret = os.getenv("INFISICAL_CLIENT_SECRET", "your_infisical_client_secret")

client = InfisicalSDKClient(host="https://app.infisical.com")

client.auth.universal_auth.login(
    client_id=client_id,
    client_secret=client_secret
)

print("SUCCESS")