import pymilvus
from app import database


def create_language_model(model_name: str, num_dims: int):
    """Create a new language model in milvus by adding a record to the meta
    collection, and createing a new collection for the language model"""
    meta_collection = database.create_or_get_model_metadata()
    meta_collection.insert(
        [
            model_name,
            num_dims,
        ]
    )
    model_collection = __create_language_model_colection__(model_name, num_dims)
    return model_collection


def delete_language_model(model_name: str):
    """Delete a language model from milvus by removing the record from the meta
    collection, and deleting the collection for the language model"""
    meta_collection = database.create_or_get_model_metadata()
    meta_collection.delete_records([model_name])
    model_collection = pymilvus.Collection(
        name=f"model.{model_name}", using=database.MILVUS_ALIAS
    )
    model_collection.drop()
    return model_collection


def __create_language_model_colection__(model_name: str, num_dims: int):
    """Create a new collection for a language model"""
    model_name = f"model.{model_name}"
    if pymilvus.utility.has_collection(model_name, using=database.MILVUS_ALIAS):
        raise ValueError(f"Collection {model_name} already exists")
    word_field = pymilvus.FieldSchema(
        name="word",
        dtype=pymilvus.DataType.VARCHAR,
        max_length=128,
    )
    embedding_field = pymilvus.FieldSchema(
        name="embedding",
        dtype=pymilvus.DataType.FLOAT_VECTOR,
        dim=num_dims,
    )
    schema = pymilvus.CollectionSchema(
        fields=[word_field, embedding_field],
        description=f"Language model {model_name} d={num_dims}",
    )
    collection = pymilvus.Collection(
        name=model_name,
        schema=schema,
        using=database.MILVUS_ALIAS,
    )
    return collection


def insert_embedding(model_name: str, word: str, embedding: list[float]):
    """Insert a word embedding into a language model"""
    model_collection = pymilvus.Collection(
        name=f"model.{model_name}", using=database.MILVUS_ALIAS
    )
    model_collection.insert(
        [
            word,
            embedding,
        ]
    )
    return model_collection

def remove_embeddings(model_name: str, words: str):
    """Remove a word embedding from a language model"""
    model_collection = pymilvus.Collection(
        name=f"model.{model_name}", using=database.MILVUS_ALIAS
    )
    words = ",".join(words)
    expr = f"word in ({words})"
    model_collection.delete(expr)
    return model_collection
