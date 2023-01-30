import random

import fastapi

app = fastapi.FastAPI()


@app.get("/")
def healthcheck():
    return {"status": "healthy"}


### Language Model Router

language_model_router = fastapi.APIRouter(
    prefix="/models",
    tags=["language models"],
)


@language_model_router.get("/")
def list_models():
    """List all available language models"""
    return {"models": ["en_core_web_sm"]}


@language_model_router.get("/{model_name}")
def view_model(model_name: str):
    """View a language models info"""
    return {"model": model_name}


@language_model_router.put("/{model_name}")
def put_model(model_name: str):
    """Add a language model"""
    # create a new collection for this language model
    return {"added": model_name}


@language_model_router.delete("/{model_name}")
def delete_model(model_name: str):
    """Delete a language model"""
    # delete all vectors for this model
    return {"deleted": model_name}


@language_model_router.get("/{model_name}/word/{word}")
def view_embeddings(model_name: str, word: str):
    """View a language models embeddings"""
    return {
        word: [random.random() for _ in range(300)],
        "word": word,
        "model": model_name,
    }


@language_model_router.put("/{model_name}/{word}")
def put_embedding_for_model(model_name: str, word: str, embedding: list):
    """Put an embedding for a specific language model"""
    return {"model": model_name, "word": word, "msg": "added"}


@language_model_router.delete("/{model_name}/{word}")
def remove_embdedding_for_model(model_name: str, word: str):
    """Remove an embedding for a specific language model"""
    return {"msg": "removed", "word": word, "model": model_name}


app.include_router(language_model_router)
