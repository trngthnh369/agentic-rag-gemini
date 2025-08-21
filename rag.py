# rag.py
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import os
import numpy as np
import chromadb
import json
import re
from typing import Dict, List, Any, Tuple
from enum import Enum
from web_search import WebSearcher

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

chroma_client = chromadb.PersistentClient("db")
collection_name = 'products'

class InformationSource(Enum):
    VECTOR_DATABASE = "vector_database"
    SHOP_INFO = "shop_info"
    INTERNET = "internet"
    NONE = "none"

class QueryProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def rewrite_query(self, original_query: str) -> str:
        """Step 1: Rewrite the original query for better understanding"""
        prompt = f"""
        Bạn là một chuyên gia xử lý câu hỏi. Hãy viết lại câu hỏi sau để làm cho nó rõ ràng, cụ thể và dễ hiểu hơn:

        Câu hỏi gốc: "{original_query}"

        Yêu cầu:
        1. Giữ nguyên ý nghĩa chính
        2. Làm rõ các từ khóa quan trọng
        3. Bổ sung ngữ cảnh nếu cần
        4. Chỉ trả về câu hỏi đã được viết lại, không giải thích

        Câu hỏi đã viết lại:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in query rewriting: {e}")
            return original_query

    def need_additional_info(self, query: str) -> bool:
        """Step 2: Determine if additional information is needed"""
        prompt = f"""
        Phân tích câu hỏi sau và quyết định xem có cần thêm thông tin bổ sung để trả lời tốt hay không:

        Câu hỏi: "{query}"

        Hãy trả lời CHÍNH XÁC một trong hai từ:
        - "YES" nếu cần thêm thông tin từ cơ sở dữ liệu, internet hoặc nguồn khác
        - "NO" nếu có thể trả lời trực tiếp dựa trên kiến thức chung

        Trả lời:
        """
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip().upper()
            return answer == "YES"
        except Exception as e:
            print(f"Error in need_additional_info: {e}")
            return True

    def determine_information_source(self, query: str) -> List[InformationSource]:
        """Step 3: Determine which information sources to use"""
        prompt = f"""
        Phân tích câu hỏi sau và xác định nguồn thông tin nào cần thiết để trả lời:

        Câu hỏi: "{query}"

        Các nguồn thông tin có sẵn:
        1. vector_database - Chứa thông tin sản phẩm, giá cả, thông số kỹ thuật
        2. shop_info - Chứa thông tin cửa hàng, địa chỉ, giờ mở cửa
        3. internet - Tìm kiếm thông tin mới nhất từ internet
        4. none - Không cần nguồn thông tin bổ sung

        Hãy trả lời bằng JSON format với key "sources" chứa danh sách các nguồn cần thiết:
        Ví dụ: {{"sources": ["vector_database", "shop_info"]}}

        Trả lời:
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip())
            sources = []
            for source in result.get("sources", []):
                try:
                    sources.append(InformationSource(source))
                except ValueError:
                    continue
            return sources if sources else [InformationSource.NONE]
        except Exception as e:
            print(f"Error in determine_information_source: {e}")
            return [InformationSource.VECTOR_DATABASE]

    def evaluate_response_quality(self, query: str, response: str) -> bool:
        """Step 4: Evaluate if the response adequately answers the query"""
        prompt = f"""
        Đánh giá xem câu trả lời sau có trả lời tốt và liên quan đến câu hỏi không:

        Câu hỏi: "{query}"
        Câu trả lời: "{response}"

        Tiêu chí đánh giá:
        1. Có trả lời đúng trọng tâm câu hỏi không?
        2. Thông tin có chính xác và đầy đủ không?
        3. Có liên quan trực tiếp đến câu hỏi không?

        Hãy trả lời CHÍNH XÁC một trong hai từ:
        - "YES" nếu câu trả lời tốt và đầy đủ
        - "NO" nếu câu trả lời chưa tốt hoặc thiếu thông tin

        Trả lời:
        """
        
        try:
            response_eval = self.model.generate_content(prompt)
            answer = response_eval.text.strip().upper()
            return answer == "YES"
        except Exception as e:
            print(f"Error in evaluate_response_quality: {e}")
            return True

class RAG:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.web_searcher = WebSearcher()
        self.max_iterations = 3
        
    def get_embedding(self, text: str) -> list[float]:
        """Generate embeddings using Gemini"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def retrieve_from_vector_db(self, query: str) -> str:
        """Retrieve information from vector database"""
        try:
            collection = chroma_client.get_collection(name=collection_name)
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return "Không thể tạo embedding cho câu truy vấn."
            
            query_embedding = np.array(query_embedding)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            search_results = collection.query(
                query_embeddings=query_embedding.tolist(), 
                n_results=5
            )

            metadatas = search_results.get('metadatas', [])
            search_result = ""
            
            for i, metadata_list in enumerate(metadatas):
                if isinstance(metadata_list, list):
                    for j, metadata in enumerate(metadata_list):
                        if isinstance(metadata, dict):
                            combined_text = metadata.get('information', 'No text available').strip()
                            search_result += f"{i+1}). {combined_text}\n\n"
                            
            return search_result if search_result else "Không tìm thấy thông tin liên quan."
            
        except Exception as e:
            print(f"Error in retrieve_from_vector_db: {e}")
            return "Lỗi khi truy xuất dữ liệu từ cơ sở dữ liệu."

    def retrieve_shop_info(self) -> str:
        """Retrieve shop information from Google Sheets"""
        try:
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

            data = sheet.get_all_records()
            return json.dumps(data, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"Error in retrieve_shop_info: {e}")
            return "Lỗi khi truy xuất thông tin cửa hàng."

    def search_internet(self, query: str) -> str:
        """Search internet for additional information using SerpAPI"""
        results = self.web_searcher.search(query)
        return self.web_searcher.format_results(results)

    def retrieve_information(self, query: str, sources: List[InformationSource]) -> str:
        """Retrieve information from specified sources"""
        retrieved_info = []
        
        for source in sources:
            if source == InformationSource.VECTOR_DATABASE:
                info = self.retrieve_from_vector_db(query)
                retrieved_info.append(f"Từ cơ sở dữ liệu sản phẩm:\n{info}")
                
            elif source == InformationSource.SHOP_INFO:
                info = self.retrieve_shop_info()
                retrieved_info.append(f"Thông tin cửa hàng:\n{info}")
                
            elif source == InformationSource.INTERNET:
                info = self.search_internet(query)
                retrieved_info.append(f"Từ internet:\n{info}")
        
        return "\n\n---\n\n".join(retrieved_info)

    def generate_response(self, query: str, context: str = "") -> str:
        """Generate final response using Gemini"""
        if context:
            prompt = f"""
            Dựa trên thông tin sau, hãy trả lời câu hỏi một cách chính xác và hữu ích:

            Thông tin tham khảo:
            {context}

            Câu hỏi: {query}

            Yêu cầu:
            1. Trả lời chính xác và đầy đủ
            2. Sử dụng thông tin từ ngữ cảnh được cung cấp
            3. Nếu không có thông tin đủ, hãy nói rõ
            4. Trả lời bằng tiếng Việt

            Câu trả lời:
            """
        else:
            prompt = f"""
            Hãy trả lời câu hỏi sau dựa trên kiến thức của bạn:

            Câu hỏi: {query}

            Yêu cầu:
            1. Trả lời chính xác và hữu ích
            2. Trả lời bằng tiếng Việt
            3. Nếu không chắc chắn, hãy nói rõ

            Câu trả lời:
            """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return "Xin lỗi, tôi không thể tạo câu trả lời lúc này."

    def process_query(self, original_query: str) -> str:
        """Main processing workflow with web search fallback"""
        current_query = original_query
        iteration = 0
        
        while iteration < self.max_iterations:
            print(f"Iteration {iteration + 1}")
            
            # Step 1: Rewrite query
            updated_query = self.query_processor.rewrite_query(current_query)
            print(f"Updated query: {updated_query}")
            
            # Step 2: Check if additional info is needed
            needs_info = self.query_processor.need_additional_info(updated_query)
            print(f"Needs additional info: {needs_info}")
            
            if not needs_info:
                # Generate response directly
                response = self.generate_response(updated_query)
            else:
                # Step 3: Determine information sources
                sources = self.query_processor.determine_information_source(updated_query)
                print(f"Information sources: {[s.value for s in sources]}")
                
                # First try database
                context = self.retrieve_information(updated_query, sources)
                
                # If no useful context found in database, try web search
                if not context or len(context.strip()) < 50:  # Adjust threshold as needed
                    print("No sufficient information in database, trying web search...")
                    web_results = self.search_internet(updated_query)
                    if web_results:
                        context = web_results
                        print("Found information from web search")
                    else:
                        print("No information found from web search")
                
                print(f"Total context length: {len(context)}")
                
                # Generate response with context
                response = self.generate_response(updated_query, context)
            
            print(f"Generated response: {response[:200]}...")
            
            # Step 4: Evaluate response quality
            is_good_response = self.query_processor.evaluate_response_quality(
                updated_query, response
            )
            print(f"Response quality good: {is_good_response}")
            
            if is_good_response:
                return response
            
            # If response is not good, update query for next iteration
            current_query = f"Câu hỏi gốc: {original_query}. Câu trả lời trước không đầy đủ: {response}. Hãy cải thiện câu hỏi."
            iteration += 1
        
        # Return the last response if max iterations reached
        return response

#  RAG functions (no longer need @function_tool decorator)
rag = RAG()

def product_rag(query: str) -> str:
    """ RAG for product information with reasoning and action"""
    print(f'---- Product RAG: {query}')
    return rag.process_query(query)

def shop_info_rag(query: str) -> str:
    """ RAG for shop information with reasoning and action"""
    print(f'---- Shop Info RAG: {query}')
    return rag.process_query(query)