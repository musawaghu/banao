"""
FastAPI Backend Application
Main API endpoints for Recipe AI App
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from contextlib import asynccontextmanager

# Import services (assuming they're in services/ directory)
# from app.services.llm_service import LLMService, RecipeRequest, GeneratedRecipe
# from app.services.rag_service import RAGService
# from app.services.image_service import ImageRecognitionService
# from app.services.youtube_service import YouTubeService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service instances (singleton pattern)
llm_service = None
rag_service = None
image_service = None
youtube_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    global llm_service, rag_service, image_service, youtube_service

    logger.info("Initializing services...")
    # Uncomment when services are imported
    # llm_service = LLMService(provider="gemini")
    # rag_service = RAGService()
    # image_service = ImageRecognitionService(provider="gemini")
    # youtube_service = YouTubeService()
    logger.info("Services initialized")

    yield

    # Shutdown
    logger.info("Shutting down services...")
    if youtube_service:
        await youtube_service.close()
    logger.info("Services shut down")


# Create FastAPI app
app = FastAPI(
    title="Recipe AI API",
    description="AI-powered recipe generation with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class IngredientDetectionResponse(BaseModel):
    ingredients: List[str]
    count: int


class RecipeGenerationRequest(BaseModel):
    ingredients: List[str] = Field(..., min_items=1, max_items=20)
    calories: int = Field(..., ge=100, le=2000)
    cuisine: str = Field(..., min_length=1)
    servings: int = Field(default=4, ge=1, le=12)
    dietary_restrictions: Optional[List[str]] = None
    difficulty: str = Field(default="any")


class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: List[str]
    calories: int
    prep_time: int
    cook_time: int
    servings: int
    cuisine: str
    difficulty: str
    tips: List[str]
    youtube_url: Optional[str] = None
    youtube_thumbnail: Optional[str] = None


class RecipeGenerationResponse(BaseModel):
    recipes: List[Recipe]
    generation_time: float


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict


# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "llm": llm_service is not None,
            "rag": rag_service is not None,
            "image": image_service is not None,
            "youtube": youtube_service is not None
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return await root()


@app.post("/api/detect-ingredients", response_model=IngredientDetectionResponse)
async def detect_ingredients(file: UploadFile = File(...)):
    """
    Detect ingredients from uploaded image

    - **file**: Image file (JPEG, PNG)
    """
    if not image_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image recognition service not available"
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    try:
        # Read image data
        image_data = await file.read()

        # Validate size (max 5MB)
        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image size must be less than 5MB"
            )

        # Detect ingredients
        ingredients = await image_service.detect_ingredients(image_data)

        if not ingredients:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No ingredients detected in image"
            )

        logger.info(f"Detected {len(ingredients)} ingredients")

        return {
            "ingredients": ingredients,
            "count": len(ingredients)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingredient detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process image"
        )


@app.post("/api/generate-recipes", response_model=RecipeGenerationResponse)
async def generate_recipes(request: RecipeGenerationRequest):
    """
    Generate 3 recipes based on ingredients and preferences

    - **ingredients**: List of available ingredients
    - **calories**: Target calories per serving
    - **cuisine**: Desired cuisine type
    - **servings**: Number of servings (default: 4)
    - **dietary_restrictions**: Optional dietary restrictions
    - **difficulty**: Recipe difficulty (default: "any")
    """
    if not llm_service or not rag_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recipe generation service not available"
        )

    try:
        import time
        start_time = time.time()

        # Create internal request object
        # Uncomment when services are imported
        # recipe_request = RecipeRequest(
        #     ingredients=request.ingredients,
        #     calories=request.calories,
        #     cuisine=request.cuisine,
        #     servings=request.servings,
        #     dietary_restrictions=request.dietary_restrictions,
        #     difficulty=request.difficulty
        # )

        # Mock data for now
        logger.info(f"Generating recipes for: {request.ingredients}")

        # Step 1: Search for similar recipes using RAG
        # similar_recipes = await rag_service.search_similar_recipes(
        #     recipe_request,
        #     top_k=10
        # )

        # if not similar_recipes:
        #     raise HTTPException(
        #         status_code=status.HTTP_404_NOT_FOUND,
        #         detail="No similar recipes found for this combination"
        #     )

        # Step 2: Generate recipes using LLM
        # generated_recipes = await llm_service.generate_recipes(
        #     recipe_request,
        #     similar_recipes
        # )

        # Step 3: Find YouTube videos for each recipe
        # for recipe in generated_recipes:
        #     video_data = await youtube_service.search_recipe_video(recipe.name)
        #     if video_data:
        #         recipe.youtube_url = video_data["url"]
        #         recipe.youtube_thumbnail = video_data["thumbnail"]

        # Mock response for testing
        mock_recipes = [
            {
                "name": f"Delicious {request.cuisine} Dish {i + 1}",
                "ingredients": request.ingredients + ["olive oil", "salt", "pepper"],
                "instructions": [
                    "Prepare all ingredients",
                    "Cook according to traditional methods",
                    "Season to taste",
                    "Serve hot"
                ],
                "calories": request.calories,
                "prep_time": 15,
                "cook_time": 30,
                "servings": request.servings,
                "cuisine": request.cuisine,
                "difficulty": "Medium",
                "tips": ["Use fresh ingredients", "Adjust seasoning to taste"],
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "youtube_thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
            }
            for i in range(3)
        ]

        generation_time = time.time() - start_time

        logger.info(f"Generated {len(mock_recipes)} recipes in {generation_time:.2f}s")

        return {
            "recipes": mock_recipes,
            "generation_time": generation_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recipe generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recipes: {str(e)}"
        )


@app.get("/api/cuisines")
async def get_cuisines():
    """Get list of supported cuisines"""
    cuisines = [
        "Italian", "Chinese", "Japanese", "Mexican", "Indian",
        "Thai", "French", "Greek", "Spanish", "Korean",
        "Vietnamese", "Turkish", "Lebanese", "American", "Mediterranean"
    ]
    return {"cuisines": sorted(cuisines)}


@app.get("/api/dietary-restrictions")
async def get_dietary_restrictions():
    """Get list of dietary restrictions"""
    restrictions = [
        "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free",
        "Nut-Free", "Low-Carb", "Keto", "Paleo", "Halal", "Kosher"
    ]
    return {"restrictions": sorted(restrictions)}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )