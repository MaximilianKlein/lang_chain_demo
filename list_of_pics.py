from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_vertexai import VertexAI

class ListOfPictures(BaseModel):
    asksToDraw : bool = Field(description="Whether the user asks to draw something. Only true of false")
    thingsToDraw : list[str] = Field(description="List of things the user wants to draw")

parser = PydanticOutputParser(pydantic_object=ListOfPictures)

model = VertexAI(model_name="gemini-pro")

prompt = PromptTemplate(
    template="Please extract everything that needs to be drawn from the following prompt. It might contain more complex tasts like 'draw a flower in a field' that describe the thing a bit."
           "\n{format_instructions}\n{prompt}\n",
    input_variables=["prompt"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

prompt = input("Please enter your prompt: ")

res = chain.invoke({"prompt": prompt})

print(res)

def combine_ascii_images_centered_list(ascii_images, spacing=8):
    # Split each image into lines
    split_images = [img.split('\n') for img in ascii_images]
    max_height = max(len(img) for img in split_images)
    
    # Calculate the max width for each image to ensure consistent padding
    max_widths = [max(len(line) for line in img) for img in split_images]
    
    # Center each image vertically and ensure each line has uniform width
    centered_images = []
    for img, max_width in zip(split_images, max_widths):
        top_padding = (max_height - len(img)) // 2
        bottom_padding = max_height - len(img) - top_padding
        padded_img = [' ' * max_width for _ in range(top_padding)] + \
                     [line.ljust(max_width) for line in img] + \
                     [' ' * max_width for _ in range(bottom_padding)]
        centered_images.append(padded_img)
    
    # Combine the images side by side with specified spacing
    combined_lines = []
    for i in range(max_height):
        combined_line = (' ' * spacing).join(centered_images[j][i] for j in range(len(centered_images)))
        combined_lines.append(combined_line)
    
    return '\n'.join(combined_lines)


asciiArts = ["*"]
if res.asksToDraw == True:
    for thing in res.thingsToDraw:
        drawing = model.invoke(f"Draw a '{thing}' in ASCII art without any further text only the raw output no ` and keep it simple. It shouldn't be higher than 20 lines")
        asciiArts.append(drawing)

asciiArts.append("*")

onePic = combine_ascii_images_centered_list(asciiArts)

print(onePic)
