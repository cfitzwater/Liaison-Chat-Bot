import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(override=True)
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(f"Testing connection to: {url}")
client = create_client(url, key)
response = client.table("liaison_library").select("count", count="exact").execute()
print(f"Success! Found {response.count} rows in the liaison_library table.")