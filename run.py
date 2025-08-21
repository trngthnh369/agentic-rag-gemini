#!/usr/bin/env python3
# run.py - Simple script to run the Gemini RAG system

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        ".env",
        "agentic-rag-469610-2993c19c6c9b.json", 
        "hoanghamobile.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure these files exist before running.")
        return False
    
    return True

def check_env_variables():
    """Check if required environment variables are set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        print("Please add your Gemini API key to .env file:")
        print("GEMINI_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ Gemini API key found: {gemini_key[:10]}...")
    return True

def setup_database():
    """Setup the database if needed"""
    if not Path("db").exists():
        print("🔧 Setting up database...")
        try:
            subprocess.run([sys.executable, "setup.py"], check=True)
            print("✅ Database setup completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Database setup failed: {e}")
            return False
    else:
        print("✅ Database already exists")
    
    return True

def start_server():
    """Start the Gemini RAG server"""
    print("🚀 Starting Gemini RAG server...")
    try:
        subprocess.run([sys.executable, "serve.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

def start_ui():
    """Start the Streamlit UI"""
    print("🎨 Starting Streamlit UI...")
    try:
        subprocess.run(["streamlit", "run", "client.py"])
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"❌ UI error: {e}")

def main():
    print("🤖 Gemini RAG System Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if not check_env_variables():
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        sys.exit(1)
    
    # Ask user what to run
    print("\nWhat would you like to run?")
    print("1. Server only (API)")
    print("2. UI only (Streamlit)")
    print("3. Both (server + UI)")
    print("4. Test Gemini connection")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        start_server()
    elif choice == "2":
        start_ui()
    elif choice == "3":
        print("Starting both server and UI...")
        print("Server will start first, then UI will open in browser")
        
        # Start server in background
        import threading
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Start UI
        start_ui()
    elif choice == "4":
        test_gemini()
    else:
        print("Invalid choice")
        sys.exit(1)

def test_gemini():
    """Test Gemini API connection"""
    print("🧪 Testing Gemini connection...")
    
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say 'Hello from Gemini!'")
        
        print("✅ Gemini connection successful!")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")

if __name__ == "__main__":
    main()