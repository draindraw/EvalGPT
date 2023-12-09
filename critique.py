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
        template = "Please provide a comprehensive critique for the following startup idea, analyzing both its strengths and weaknesses. Be thorough in your evaluation, considering aspects such as market viability, potential user adoption, scalability, and differentiation from existing solutions. Additionally, identify specific drawbacks and strong points, offering constructive feedback on how the idea could be improved or expanded. Please ensure your analysis covers key areas such as the target audience, revenue model, user experience, and potential challenges in implementation. Your insights should be detailed and actionable, aiming to guide the startup toward optimal development and success. No matter the size of the idea, provide a thorough response with specific tips and recommendations to enhance its overall potential and viability in the market. Here is the startup idea : {idea}"

    )

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'output')

    response = title_chain({'idea' : data.idea})

    return {"content": response["output"]}

