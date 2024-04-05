from langchain_google_vertexai import VertexAI

model = VertexAI(model_name="gemini-pro")

prompt = input("Please enter your prompt: ")
message = ("For the following prompt return a json that contains "
           "one field: 'asksToDraw' which is either true or false "
           "and another field 'thingsToDraw' which is an array containing the things that the user wants us to draw. This can be anything indicated by 'draw this' or similar"
           "\nFor example: 'draw a flower in a field' would yield "
           "'{\"asksToDraw\": true, \"thingsToDraw\": [\"flower in a field\"]}"
           "\nAnother example: 'how many people live in portugal' would yield "
           "'{\"asksToDraw\": false, \"thingsToDraw\": []}"
           "\n\nPROMPT:\n" + prompt
            )

res = model.invoke(message)

print(res)

