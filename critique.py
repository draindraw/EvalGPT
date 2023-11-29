from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

os.environ['GOOGLE_API_KEY'] = "AIzaSyAbhkn4KQxk8lBNtVEF3sNXV1e47SzO2Ic"

llm = GooglePalm(temperature = 0.3)


class InputData(BaseModel):
    idea: str

@app.post("/predict")
async def generate_text(data: InputData):

    title_template = PromptTemplate(
        input_variables = ['idea'],
        template = "Act as a harsh critique for the following startup idea and provide feedback like what are the drawbacks and the strong points, also give tips to improve or add to the idea and based on the current market trends and different aspects provide a rough estimate of evaluation(in INR) : {idea} "

    )

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'output')

    response = title_chain({'idea' : data.idea})

    return {"content": response["output"]}

