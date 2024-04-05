from langchain_google_vertexai import VertexAI

model = VertexAI(model_name="gemini-pro")

message = "What are some of the pros and cons of Python as a programming language?"
res = model.invoke(message)

print(res)

