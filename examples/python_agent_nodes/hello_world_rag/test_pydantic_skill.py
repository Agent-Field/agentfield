"""
Demonstration: Skills now support both Pydantic models AND plain parameters

This script shows that the fix enables:
1. Using Pydantic models for better validation (NEW)
2. Using plain parameters for simplicity (EXISTING - backward compatible)
"""

from typing import Optional
from pydantic import BaseModel
from agentfield import Agent


# Define Pydantic models
class IngestRequest(BaseModel):
    """Pydantic model for document ingestion request."""

    document_id: str
    path: Optional[str] = None
    text: Optional[str] = None


class IngestResult(BaseModel):
    """Result of document ingestion."""

    document_id: str
    chunks_processed: int
    status: str


# Create agent
app = Agent(
    node_id="pydantic-demo",
    agentfield_server="http://localhost:8080",
)


# Approach 1: Using Pydantic model (NEW - now works with the fix!)
@app.skill()
async def ingest_with_model(request: IngestRequest) -> IngestResult:
    """
    Skill using Pydantic model for input validation.

    Benefits:
    - Automatic validation of required fields
    - Type checking at runtime
    - Clear schema definition
    - FastAPI-like developer experience
    """
    print("✅ Pydantic model approach:")
    print(f"   Type: {type(request)}")
    print(f"   document_id: {request.document_id}")
    print(f"   path: {request.path}")
    print(f"   text: {request.text}")

    # Simulate processing
    chunks = 5 if request.text or request.path else 0

    return IngestResult(
        document_id=request.document_id, chunks_processed=chunks, status="success"
    )


# Approach 2: Using plain parameters (EXISTING - backward compatible)
@app.skill()
async def ingest_with_params(
    document_id: str, path: Optional[str] = None, text: Optional[str] = None
) -> IngestResult:
    """
    Skill using plain parameters.

    Benefits:
    - Simple and straightforward
    - No need to define models for simple cases
    - Backward compatible with existing code
    """
    print("✅ Plain parameters approach:")
    print(f"   document_id: {document_id}")
    print(f"   path: {path}")
    print(f"   text: {text}")

    # Simulate processing
    chunks = 5 if text or path else 0

    return IngestResult(
        document_id=document_id, chunks_processed=chunks, status="success"
    )


if __name__ == "__main__":
    print("🎉 Skills now support Pydantic models!")
    print("=" * 70)
    print("\nRegistered skills:")
    print("  1️⃣  ingest_with_model(request: IngestRequest)")
    print("     → Uses Pydantic model for validation")
    print()
    print("  2️⃣  ingest_with_params(document_id, path, text)")
    print("     → Uses plain parameters (backward compatible)")
    print("=" * 70)
    print("\nStarting agent server...")
    print("You can test with:\n")
    print("  af run pydantic-demo.ingest_with_model \\")
    print('    --input \'{"document_id":"doc1","path":"/tmp/test.txt"}\'')
    print()
    print("  af run pydantic-demo.ingest_with_params \\")
    print('    --input \'{"document_id":"doc2","text":"Hello"}\'')
    print()

    app.run(auto_port=True)
