import yaml
import os
from smolagents import GradioUI, CodeAgent, InferenceClientModel, PythonInterpreterTool
import pandas as pd
import requests
from io import BytesIO
from huggingface_hub import InferenceClient

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.answer_question import SimpleTool as AnswerQuestion
from tools.web_search import DuckDuckGoSearchTool as WebSearch
from tools.visit_webpage import VisitWebpageTool as VisitWebpage
from tools.final_answer import FinalAnswerTool as FinalAnswer

class GAIAAgent:
    def __init__(self):

        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

        model = InferenceClientModel(
                model_id="anthropic/claude-3.5-sonnet-20241022", 
                provider="anthropic",
                token=os.getenv("ANTHROPIC_API_KEY")
        )
        
        with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
            prompt_templates = yaml.safe_load(stream)
        
        answer_question = AnswerQuestion()
        web_search = WebSearch()
        visit_webpage = VisitWebpage()
        final_answer = FinalAnswer()
        python_interpreter = PythonInterpreterTool()
        
        self.agent = CodeAgent(
            model=model,
            tools=[
                answer_question, 
                web_search, 
                visit_webpage, 
                python_interpreter,
                final_answer,
            ],
            managed_agents=[],
            max_steps=15,
            verbosity_level=1,
            grammar=None,
            planning_interval=3,
            name='gaia_agent',
            description='Agent for answering GAIA evaluation questions',
            executor_type='local',
            executor_kwargs={},
            max_print_outputs_length=None,
            prompt_templates=prompt_templates
        )
    
    def run(self, question: str) -> str:
        return self.agent.run(question)

agent = GAIAAgent()

if __name__ == "__main__":
    GradioUI(agent.agent).launch()