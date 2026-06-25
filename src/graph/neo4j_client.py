import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME,NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()

    def execute_write(self, query:str, parameters: dict = None):
        parameters = parameters or {}

        with self.driver.session() as session:
            return session.execute_write(
                lambda tx: list(tx.run(query,parameters))
            )
    
    def execute_read(self, query: str, parameters: dict = None):
        parameters = parameters or {}
        with self.driver.session() as session:
            return session.execute_read(
                lambda tx: list(tx.run(query,parameters))
            )

def test_connection():
    client = Neo4jClient()

    try: 
        result = client.execute_read("RETURN 'Neo4j connected' AS message")
        print(f"{result}\n")
        print(result[0]["message"])
    finally:
        client.close()

if __name__ == "__main__":
    test_connection()