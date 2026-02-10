import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

print("--- Database Diagnostic ---")
collections = client.list_collections()
print(f"Total Collections found: {len(collections)}")

for coll in collections:
    print(f"\nCollection Name: {coll.name}")
    count = coll.count()
    print(f"Total Items: {count}")
    
    if count > 0:
        sample = coll.get(limit=1)
        print(f"Sample Metadata Keys: {sample['metadatas'][0].keys() if sample['metadatas'] else 'None'}")
        print(f"Sample ID: {sample['ids'][0]}")