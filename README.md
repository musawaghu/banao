#  banao — Your Personal AI-Powered Recipe Creator

banao is an intelligent iOS app that helps you turn ingredients into delicious recipes using cutting-edge AI.  
Simply tell banao:
- What ingredients you have 🥦  
- How many calories you want 🍽️  
- What cuisine you’re craving 🌍  

…and banao will generate **3 smart, healthy, and personalized recipes** for you — complete with ingredients, instructions, and optional YouTube cooking videos!  

---

## 🚀 Features

✅ **AI Recipe Generation**
- Uses a fine-tuned LLM (OpenAI GPT / Mistral / Llama 3) for natural, creative recipe creation.  
- Supports custom parameters like calorie limits, cuisine type, and dietary restrictions.  

✅ **RAG (Retrieval-Augmented Generation)**
- Recipes are grounded with real cookbook and recipe dataset knowledge.  
- The RAG pipeline ensures factual, high-quality recipes rather than purely generative outputs.

✅ **YouTube Integration**
- Fetches related cooking videos for the generated recipes via the YouTube Data API.

✅ **Modern iOS Design**
- Built with **SwiftUI** for a sleek, responsive, and intuitive user experience.  
- Optimized for both light and dark modes.

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | SwiftUI (iOS 17+) |
| **Backend** | FastAPI or Flask (Python) |
| **LLM** | OpenAI GPT-4, Mistral, or local Llama-3 model |
| **RAG Pipeline** | LangChain / LlamaIndex + FAISS vector store |
| **Database** | PostgreSQL / Firebase (user data, preferences, and logs) |
| **YouTube Integration** | YouTube Data API v3 |
| **Cloud Hosting** | AWS / Render / Vercel backend deployment |

---

## 🧩 System Architecture

1. **User Input (iOS App)**
   - User enters:  
     - Ingredients  
     - Target Calories  
     - Cuisine Type  
2. **Backend API (Python + FastAPI)**  
   - Receives user data  
   - Queries RAG system for relevant recipes  
   - Calls LLM to generate 3 refined, creative recipe options  
3. **YouTube Video Fetcher**  
   - Uses recipe keywords to pull top related cooking tutorials.  
4. **Response Rendered (SwiftUI)**  
   - Recipes displayed beautifully with names, steps, and embedded video thumbnails.  
