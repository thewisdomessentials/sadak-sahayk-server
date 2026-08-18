import os
from test_qdrant_source import test_query_with_sources

if __name__ == "__main__":
    # Reframing the TC-020 query to explicitly demand the legal sections and penalty citations
    test_query = """
        i got an accident by some citizen who was doing rash driving and i got minor injury he ran away now what ipc and crpc acts will be applied on him  """
    
    print("Testing TC-020 isolated query...")
    test_query_with_sources(test_query, limit=15)
