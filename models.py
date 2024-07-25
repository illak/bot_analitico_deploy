from pydantic import BaseModel

class QueryRequest(BaseModel):
    text: str
    space_name: str