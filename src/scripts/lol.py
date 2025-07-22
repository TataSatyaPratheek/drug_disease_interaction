import weaviate

def check_and_populate():
    """Quick check and populate if needed"""
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        
        # Check current data
        collections = ["Drug", "Disease", "Protein", "Relationship"]
        total_count = 0
        
        for collection_name in collections:
            if client.collections.exists(collection_name):
                collection = client.collections.get(collection_name)
                response = collection.aggregate.over_all(total_count=True)
                count = response.total_count or 0
                total_count += count
                print(f"{collection_name}: {count:,} objects")
        
        print(f"Total entities: {total_count:,}")
        
        if total_count == 0:
            print("❌ No data found. Collections exist but are empty.")
            print("🔧 Run: python -m src.scripts.migrate_to_weaviate_debug")
        else:
            print("✅ Data exists in Weaviate")
            
        client.close()
        
    except Exception as e:
        print(f"❌ Check failed: {e}")

if __name__ == "__main__":
    check_and_populate()
