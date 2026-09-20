from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url="http://localhost:6333")

client.delete_collection(
    collection_name = "data",
)

if not client.collection_exists(collection_name="data"):
    print("Creating collection")
    client.create_collection(
        collection_name = "data",
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            datatype=models.Datatype.FLOAT32,
        )
    )

def embed(sentences):
    embeddings = model.encode(sentences)

    data = []

    for i in range(0, len(sentences)):
        data.append(models.PointStruct(id=i, vector=embeddings[i], payload={"sentence": sentences[i]}))

    info = client.upsert(
        collection_name = "data",
        wait=True,
        points=data,
    )

embed([
    "The weather is lovely today.",
    "It's so sunny outside!",
    "Qter is a human friendly rubik's cube computer",
])

search_result = client.query_points(
    collection_name="data",
    query=model.encode("Twisty puzzles"),
    with_payload=True,
    limit=3
).points

print(search_result)
