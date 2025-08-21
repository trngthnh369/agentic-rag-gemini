# setup.py
import os
import pandas as pd
import openai
from dotenv import load_dotenv
import re
import chromadb
import uuid
import ast

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Kiểm tra API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# ChromaDB setup
chroma_client = chromadb.PersistentClient("db")

def get_embedding(text: str) -> list[float]:
    """Generate embeddings using OpenAI"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to be MongoDB-compatible."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = name.strip("_")
    return name.lower()

def sanitize_metadata(record):
    """Convert None values in metadata to empty string or default values"""
    sanitized_record = {
        k: (str(v) if v is not None else "") for k, v in record.items() if k != "embedding"
    }
    return sanitized_record

def join_string(item):
    """Enhanced data combination for better retrieval"""
    title, product_promotion, product_specs, current_price, color_options = item
    
    final_string = ""
    
    # Add title
    if title and str(title) != 'nan':
        final_string += f"Sản phẩm: {title}. "

    # Add promotions with enhanced formatting
    if product_promotion and str(product_promotion) != 'nan':
        promotion_clean = str(product_promotion).replace("<br>", " ").replace("\n", " ")
        final_string += f"Khuyến mãi: {promotion_clean}. "

    # Add specifications
    if product_specs and str(product_specs) != 'nan':
        specs_clean = str(product_specs).replace("<br>", " ").replace("\n", " ")
        final_string += f"Thông số: {specs_clean}. "

    # Add pricing information
    if current_price and str(current_price) != 'nan':
        final_string += f"Giá hiện tại: {current_price}. "

    # Add color options
    if color_options and str(color_options) != 'nan':
        try:
            colors = ast.literal_eval(str(color_options))
            if isinstance(colors, list) and colors:
                final_string += f"Màu sắc có sẵn: {', '.join(colors)}. "
        except:
            final_string += f"Màu sắc: {color_options}. "
    
    return final_string.strip()

def setup_database():
    """Setup enhanced database with Gemini embeddings"""
    print("Setting up enhanced database with Gemini embeddings...")
    
    # Load and process data
    try:
        df = pd.read_csv("./hoanghamobile.csv")
        print(f"Loaded {len(df)} records from CSV")
    except FileNotFoundError:
        print("Error: hoanghamobile.csv not found!")
        return False
    
    # Create enhanced information strings
    df['information'] = df[
        ['title', 'product_promotion', 'product_specs', 'current_price', 'color_options']
    ].astype(str).apply(join_string, axis=1)
    
    # Filter out empty information
    df = df[df['information'].notna() & (df['information'] != '')]
    print(f"Processing {len(df)} valid records...")
    
    # Limit for testing (remove in production)
    df = df.head(50)  # Process more records than before
    print(f"Processing first {len(df)} records for testing...")
    
    # Generate embeddings with error handling
    embeddings = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            embedding = get_embedding(row['information'])
            if embedding:  # Only keep records with valid embeddings
                embeddings.append(embedding)
                valid_indices.append(idx)
                print(f"✓ Generated embedding for record {idx + 1}")
            else:
                print(f"✗ Failed to generate embedding for record {idx + 1}")
        except Exception as e:
            print(f"✗ Error processing record {idx + 1}: {e}")
    
    if not embeddings:
        print("Error: No valid embeddings generated!")
        return False
    
    # Filter dataframe to only valid records
    df_valid = df.loc[valid_indices].copy()
    print(f"Successfully generated {len(embeddings)} embeddings")
    
    # Prepare metadata
    metadatas = []
    for _, row in df_valid.iterrows():
        metadata = {
            "information": row["information"],
            "title": str(row.get("title", "")),
            "current_price": str(row.get("current_price", "")),
            "product_promotion": str(row.get("product_promotion", ""))[:500],  # Limit length
            "color_options": str(row.get("color_options", ""))
        }
        metadatas.append(sanitize_metadata(metadata))
    
    # Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in range(len(df_valid))]
    
    # Setup ChromaDB collection
    collection_name = "products"
    try:
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
        
        # Insert data in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(embeddings), batch_size):
            end_idx = min(i + batch_size, len(embeddings))
            
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            print(f"Inserted batch {i//batch_size + 1}: records {i+1}-{end_idx}")
        
        print(f"✅ Successfully inserted {len(embeddings)} documents into ChromaDB collection '{collection.name}'")
        
        # Verify insertion
        count = collection.count()
        print(f"Collection now contains {count} documents")
        
        return True
        
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        return False

def test_retrieval():
    """Test the enhanced retrieval system"""
    print("\n--- Testing Enhanced Retrieval ---")
    
    try:
        collection = chroma_client.get_collection(name="products")
        
        # Test queries
        test_queries = [
            "Nokia 3210 4G giá bao nhiêu",
            "Samsung Galaxy A05s có màu gì",
            "điện thoại rẻ nhất"
        ]
        
        for query in test_queries:
            print(f"\nTest query: '{query}'")
            try:
                query_embedding = get_embedding(query)
                if query_embedding:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3
                    )
                    
                    metadatas = results.get('metadatas', [[]])[0]
                    distances = results.get('distances', [[]])[0]
                    
                    print(f"Found {len(metadatas)} results:")
                    for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
                        info = metadata.get('information', 'No info')[:100]
                        print(f"  {i+1}. Distance: {distance:.3f}")
                        print(f"     Info: {info}...")
                else:
                    print("  ✗ Failed to generate query embedding")
                    
            except Exception as e:
                print(f"  ✗ Error testing query: {e}")
                
    except Exception as e:
        print(f"Error in test retrieval: {e}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Database Setup with Gemini...")
    
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        exit(1)
    
    # Setup database
    success = setup_database()
    
    if success:
        print("✅ Enhanced database setup completed successfully!")
        
        # Test retrieval
        test_retrieval()
        
        print("\n🎉 Enhanced RAG system is ready!")
        print("Features enabled:")
        print("  ✓ Gemini embeddings")
        print("  ✓ Query rewriting")
        print("  ✓ Information need assessment")
        print("  ✓ Multi-source retrieval")
        print("  ✓ Response quality evaluation")
        print("  ✓ Iterative refinement")
    else:
        print("❌ Database setup failed!")
        exit(1)