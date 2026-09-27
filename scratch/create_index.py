import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

index_name = "fortune-cloud"

if index_name not in [i.name for i in pc.list_indexes()]:
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    print("Index created and ready!")
else:
    print(f"Index '{index_name}' already exists.")
