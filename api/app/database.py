import os
import pymilvus 

MILVUS_ALIAS = os.environ.get('MILVUS_ALIAS', 'default')

connection = pymilvus.connections.connect(
  alias=MILVUS_ALIAS,
  host=os.environ.get('MILVUS_HOST', 'localhost'),
  port=os.environ.get('MILVUS_PORT', '19530'),
)

def create_or_get_model_metadata():
    """Create a new collection for an index of language models"""
    model_name = pymilvus.FieldSchema(
        name="book_name", 
        dtype=pymilvus.DataType.VARCHAR, 
        max_length=256,
    )
    model_num_dims = pymilvus.FieldSchema(
        name="num_dims",
        dtype=pymilvus.DataType.INT64,
    )
    schema = pymilvus.CollectionSchema(
        fields=[model_name, model_num_dims], 
        description="Test book search"
    )
    try:
        collection = pymilvus.Collection(
            name="meta.models",
            schema=schema,
            using=MILVUS_ALIAS,
        )
    except Exception as exc:
        raise exc
    return collection
