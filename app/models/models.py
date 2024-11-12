from pydantic import BaseModel

class AnswerRequest(BaseModel):
    document_id: str
    query: str

class RetrieveRequest(BaseModel):
    document_id: str
    query: str

class UploadRequest(BaseModel):
    url: str