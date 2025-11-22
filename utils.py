from sentence_transformers import SentenceTransformer, util

encoding_model = SentenceTransformer("all-MiniLM-L6-v2")

def answer_similarity(answer_big: str, answer_small: str):
    """
    Compute semantic similarity between two answers using
    SentenceTransformers
    """

    # Compute embeddings (2 vectors)
    embeddings1 = encoding_model.encode(answer_big, convert_to_tensor=True)
    embeddings2 = encoding_model.encode(answer_small, convert_to_tensor=True)

    # Compute cosine similarities
    sim_tensor = util.cos_sim(embeddings1, embeddings2)

    # Extract cosine similarity between the two specific inputs
    similarity_score = float(sim_tensor.item())

    return similarity_score