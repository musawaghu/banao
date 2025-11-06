from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow iOS app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
# Set your API key as environment variable: export GEMINI_API_KEY='your-key-here'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#GEMINI_API_KEY = "AIzaSyCdGx9cssWau06IEb-HCWQtm9s3FmkVv6Y"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# Request model
class RecipeRequest(BaseModel):
    ingredients: str
    calories: Optional[int] = None
    cuisine: Optional[str] = None


# Response model
class RecipeResponse(BaseModel):
    recipe: str
    success: bool
    message: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Recipe Generator API is running"}


@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY not configured"
            )

        # Build the prompt
        prompt = f"Create a detailed recipe using the following:\n"
        prompt += f"Ingredients: {request.ingredients}\n"

        if request.calories:
            prompt += f"Target Calories: approximately {request.calories} calories\n"

        if request.cuisine:
            prompt += f"Cuisine Type: {request.cuisine}\n"

        prompt += "\nProvide a complete recipe with:\n"
        prompt += "- Recipe name\n"
        prompt += "- Ingredient list with measurements\n"
        prompt += "- Step-by-step cooking instructions\n"
        prompt += "- Estimated prep and cook time\n"
        prompt += "- Approximate calorie count per serving"

        # Generate content using Gemini
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)

        return RecipeResponse(
            recipe=response.text,
            success=True,
            message="Recipe generated successfully"
        )

    except Exception as e:
        return RecipeResponse(
            recipe="",
            success=False,
            message=f"Error generating recipe: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)