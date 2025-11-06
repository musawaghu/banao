import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

# --- App Initialization ---
app = FastAPI()

from typing import List

# Allow iOS app to connect to backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class RecipeRequest(BaseModel):
    ingredients: str
    calories: str
    cuisine:str
class Recipe(BaseModel):
    title: str
    description: str
class RecipeResponse(BaseModel):
    recipes: List[Recipe]


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-pro"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

@app.post("/generate_recipes", response_model=RecipeResponse)
async def generate_recipes(req: RecipeRequest):
    prompt = f"""
    Generate 3 recipes based on:
    - Ingredients: {req.ingredients}
    - Calories: {req.calories}
    - Cuisine: {req.cuisine}

    Return the result strictly in JSON format as:
    [
      {{
        "title": "Recipe name",
        "description": "Recipe description"
      }},
      ...
    ]
    """

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_URL, json=body)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Gemini API request failed")

        data = response.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Invalid Gemini response")

    # Clean up any code blocks (```json ... ```) and parse JSON
    import json, re
    text = re.sub(r"```json|```", "", text).strip()

    try:
        recipes_json = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI JSON")

    recipes = [Recipe(title=r["title"], description=r["description"]) for r in recipes_json]
    return RecipeResponse(recipes=recipes)