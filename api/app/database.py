import os
import pymilvus

MILVUS_ALIAS = os.environ.get("MILVUS_ALIAS", "default")
MODEL_META_COLLECTION_NAME = "meta_models"

connection = pymilvus.connections.connect(
    alias=MILVUS_ALIAS,
    host=os.environ.get("MILVUS_HOST", "localhost"),
    port=os.environ.get("MILVUS_PORT", "19530"),
)


def create_or_get_model_metadata():
    """Create a new collection for an index of language models"""
    if pymilvus.utility.has_collection(MODEL_META_COLLECTION_NAME, using=MILVUS_ALIAS):
        return pymilvus.Collection(name=MODEL_META_COLLECTION_NAME, using=MILVUS_ALIAS)
    else:
        model_id = pymilvus.FieldSchema(
            "model_id",
            is_primary=True,
            auto_id=True,
            dtype=pymilvus.DataType.INT64,
        )
        model_name = pymilvus.FieldSchema(
            name="model_name",
            dtype=pymilvus.DataType.VARCHAR,
            max_length=128,
        )
        model_num_dims = pymilvus.FieldSchema(
            name="num_dims",
            dtype=pymilvus.DataType.INT64,
        )
        kludge_vector_field = pymilvus.FieldSchema(
            name="kludge",
            dtype=pymilvus.DataType.FLOAT_VECTOR,
            dim=3,
        )
        schema = pymilvus.CollectionSchema(
            fields=[model_id, model_name, model_num_dims, kludge_vector_field], description="Test book search"
        )
        collection = pymilvus.Collection(
            name=MODEL_META_COLLECTION_NAME,
            schema=schema,
            using=MILVUS_ALIAS,
        )
        return collection
