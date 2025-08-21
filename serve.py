# gemini_serve.py
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from rag import RAG, InformationSource

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiAgent:
    """Base agent class using only Gemini"""
    def __init__(self, name: str, instructions: str, tools: List = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def process_message(self, message: str, context: List[Dict] = None) -> str:
        """Process a message with the agent's instructions"""
        # Prepare conversation history
        conversation_context = ""
        if context:
            for msg in context[-3:]:  # Last 3 messages for context
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Create prompt with instructions
        full_prompt = f"""
{self.instructions}

Conversation History:
{conversation_context}

Current Message: {message}

Please respond as this specialized agent. If you need to use any tools, indicate which tool and what parameters.
"""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error processing message: {str(e)}"

class GeminiManager:
    """Manager to coordinate different agents using only Gemini"""
    def __init__(self):
        self.rag = RAG()
        self.agents = self._initialize_agents()
    
    def _initialize_agents(self) -> Dict[str, GeminiAgent]:
        """Initialize all agents with their instructions"""
        
        manager_instructions = """
You are the manager of specialized agents for a Vietnamese mobile phone store. Your role is to:

1. Analyze user requests and determine which specialized agent can best handle them
2. Delegate tasks to the appropriate agent (product or shop_information)
3. Process responses and provide comprehensive answers

AVAILABLE AGENTS:
- product: Use for questions about product details, availability, pricing, features, and specifications
- shop_information: Use for questions about store location, opening hours, contact information, and store policies

PROCESS:
1. Analyze the user query to determine if it's about products or shop information
2. If it's about products (giá, sản phẩm, điện thoại, Nokia, Samsung, etc.), use "PRODUCT_AGENT"
3. If it's about shop info (địa chỉ, giờ mở cửa, cửa hàng, liên hệ, etc.), use "SHOP_AGENT"
4. If mixed or unclear, use "PRODUCT_AGENT" as default

Respond with JSON format:
{
  "selected_agent": "PRODUCT_AGENT" or "SHOP_AGENT",
  "reasoning": "explanation of why this agent was selected",
  "processed_query": "refined version of the query for the agent"
}
"""
        
        product_instructions = """
You are a product assistant for a Vietnamese mobile phone store. You specialize in:

- Product information (specifications, features, pricing)
- Product comparisons
- Availability and stock information
- Promotions and discounts
- Product recommendations

When you receive a query, you should:
1. Understand what product information the user needs
2. Use the RAG system to retrieve relevant product data
3. Provide comprehensive, accurate answers in Vietnamese
4. Include specific details like pricing, specifications, and availability

Always be helpful and provide complete information about products.
"""
        
        shop_instructions = """
You are a shop information assistant for a Vietnamese mobile phone store. You specialize in:

- Store locations and addresses
- Opening hours and schedules
- Contact information
- Store policies and services
- Branch information

When you receive a query, you should:
1. Understand what shop information the user needs
2. Use the RAG system to retrieve current shop data
3. Provide accurate, up-to-date information in Vietnamese
4. Be helpful with directions, hours, and contact details

Always provide complete and current store information.
"""
        
        return {
            "manager": GeminiAgent("manager", manager_instructions),
            "product": GeminiAgent("product", product_instructions),
            "shop_information": GeminiAgent("shop_information", shop_instructions)
        }
    
    def route_query(self, query: str, context: List[Dict] = None) -> Dict[str, str]:
        """Route query to appropriate agent"""
        try:
            # Use manager to determine routing
            manager_response = self.agents["manager"].process_message(query, context)
            
            # Try to parse JSON response
            try:
                routing_decision = json.loads(manager_response)
                selected_agent = routing_decision.get("selected_agent", "PRODUCT_AGENT")
                processed_query = routing_decision.get("processed_query", query)
            except json.JSONDecodeError:
                # Fallback: simple keyword-based routing
                shop_keywords = ["địa chỉ", "cửa hàng", "giờ mở", "liên hệ", "chi nhánh", "location", "address", "hours"]
                if any(keyword in query.lower() for keyword in shop_keywords):
                    selected_agent = "SHOP_AGENT"
                else:
                    selected_agent = "PRODUCT_AGENT"
                processed_query = query
            
            return {
                "agent": selected_agent,
                "query": processed_query,
                "original_query": query
            }
            
        except Exception as e:
            # Default fallback
            return {
                "agent": "PRODUCT_AGENT",
                "query": query,
                "original_query": query
            }
    
    def process_with_agent(self, agent_type: str, query: str, context: List[Dict] = None) -> str:
        """Process query with specific agent and RAG"""
        try:
            if agent_type == "PRODUCT_AGENT":
                # Use RAG for product information
                rag_result = self.rag.process_query(query)

                # Process with product agent
                product_prompt = f"""
Based on the following product information retrieved from our database, please provide a comprehensive answer to the user's question.

Retrieved Information:
{rag_result}

User Question: {query}

Please provide a helpful, accurate response in Vietnamese. Include specific details like pricing, specifications, and availability when available.
"""
                response = self.agents["product"].model.generate_content(product_prompt)
                return response.text.strip()
                
            elif agent_type == "SHOP_AGENT":
                # Use shop information RAG
                try:
                    # Get shop information from Google Sheets
                    import gspread
                    from oauth2client.service_account import ServiceAccountCredentials

                    scope = ['https://spreadsheets.google.com/feeds',
                            'https://www.googleapis.com/auth/drive']
                    credentials = ServiceAccountCredentials.from_json_keyfile_name(
                        'mles-class-12c1216b7303.json', scope
                    )
                    client = gspread.authorize(credentials)
                    sheet = client.open_by_url(
                        'https://docs.google.com/spreadsheets/d/1mOkgLyo1oedOG1nlvoSHpqK9-fTFzE9ysLuKob9TXlg'
                    ).sheet1
                    shop_data = sheet.get_all_records()
                    shop_info = json.dumps(shop_data, ensure_ascii=False, indent=2)
                except Exception as e:
                    shop_info = "Không thể lấy thông tin cửa hàng từ database."
                
                shop_prompt = f"""
Based on the following shop information from our database, please provide a comprehensive answer to the user's question.

Shop Information:
{shop_info}

User Question: {query}

Please provide a helpful, accurate response in Vietnamese. Include specific details like addresses, opening hours, and contact information when available.
"""
                response = self.agents["shop_information"].model.generate_content(shop_prompt)
                return response.text.strip()
            
            else:
                return "Xin lỗi, tôi không thể xác định loại câu hỏi này."
                
        except Exception as e:
            return f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

# Global manager instance
gemini_manager = GeminiManager()

# Conversation history storage
conversation_history = {}

@app.route("/chat", methods=["POST"])
def gemini_chat():
    """Chat endpoint using only Gemini (no agents framework)"""
    data = request.json
    query = data.get("message", "")
    thread_id = data.get("thread_id", str(uuid.uuid4()))

    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    # Initialize conversation history for new threads
    if thread_id not in conversation_history:
        conversation_history[thread_id] = []

    try:
        # Get conversation context
        context = conversation_history[thread_id]
        
        # Route query to appropriate agent
        routing_info = gemini_manager.route_query(query, context)
        print(f"Routing: {routing_info}")
        
        # Process with selected agent
        response = gemini_manager.process_with_agent(
            routing_info["agent"], 
            routing_info["query"], 
            context
        )
        
        # Update conversation history
        conversation_history[thread_id].append({"role": "user", "content": query})
        conversation_history[thread_id].append({"role": "assistant", "content": response})
        
        # Keep only last 10 messages to manage memory
        if len(conversation_history[thread_id]) > 10:
            conversation_history[thread_id] = conversation_history[thread_id][-10:]
        
        print(f"Processed query with {routing_info['agent']}: {query}")
        
        return jsonify({
            "role": "assistant",
            "content": response,
            "agent_used": routing_info["agent"],
            "thread_id": thread_id,
            "gemini_powered": True
        })

    except Exception as e:
        print(f"Error in gemini chat processing: {e}")
        return jsonify({
            "error": "Internal server error during processing",
            "details": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "ai_provider": "Google Gemini",
        "agents": ["manager", "product", "shop_information"],
        "rag": rue
    })

@app.route("/stats", methods=["GET"])
def conversation_stats():
    """Get conversation statistics"""
    return jsonify({
        "active_threads": len(conversation_history),
        "total_messages": sum(len(messages) for messages in conversation_history.values()),
        "ai_provider": "Google Gemini",
        "features": [
            "Query routing",
            "RAG retrieval", 
            "Multi-agent simulation",
            "Conversation memory",
            "Real-time shop data"
        ]
    })

@app.route("/test", methods=["GET"])
def test_gemini():
    """Test Gemini connection"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello, respond with 'Gemini connected successfully!'")
        return jsonify({
            "status": "success",
            "response": response.text.strip(),
            "gemini_working": True
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "gemini_working": False
        }), 500

if __name__ == "__main__":
    print("🚀 Starting Pure Gemini RAG Server...")
    print("✅ No OpenAI API key required")
    print("✅ Powered by Google Gemini")
    print("✅ RAG with intelligent routing")
    
    # Test Gemini connection on startup
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        test_response = model.generate_content("Test connection")
        print("✅ Gemini connection successful")
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        print("Please check your GEMINI_API_KEY in .env file")
    
    app.run(host="0.0.0.0", port=5001, debug=True)