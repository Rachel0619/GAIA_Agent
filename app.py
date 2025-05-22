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

# from tools.youtube_video_analysis import YouTubeVideoAnalysisTool
# from tools.file_reader import FileReaderTool
# from tools.code_executor import CodeExecutorTool
# from tools.image_analyzer import ImageAnalyzerTool
# from tools.chess_analyzer import ChessAnlyzerTool
# from tools.audio_transcriber import AudioTranscriberTool
# from tools.excel_analyzer import ExcelAnalyzerTool
# from tools.wikipedia_search import WikipediaSearchTool

# Initialize tools
answer_question = AnswerQuestion()
web_search = WebSearch()
visit_webpage = VisitWebpage()
final_answer = FinalAnswer()
python_interpreter = PythonInterpreterTool()

# youtube_analyzer = YouTubeVideoAnalysisTool()
# file_reader = FileReaderTool()
# code_executor = CodeExecutorTool()
# image_analyzer = ImageAnalyzerTool()
# chess_analyzer = ChessAnalyzerTool()
# audio_transcriber = AudioTranscriberTool()
# excel_analyzer = ExcelAnalyzerTool()
# wikipedia_search = WikipediaSearchTool()

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct", 
    provider="together")

with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent_submission_agent = CodeAgent(
    model=model,
    tools=[
        answer_question, 
        web_search, 
        visit_webpage, 
        python_interpreter,
        final_answer,
        # youtube_analyzer,
        # file_reader,
        # code_executor,
        # image_analyzer,
        # chess_analyzer,
        # audio_transcriber,
        # excel_analyzer,
        # wikipedia_search
    ],
    managed_agents=[],
    max_steps=10,  # Increase steps as some questions need multiple tools
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name='submission_agent',
    description='Agent for answering GAIA evaluation questions',
    executor_type='local',
    executor_kwargs={},
    max_print_outputs_length=None,
    prompt_templates=prompt_templates
)

if __name__ == "__main__":
    GradioUI(agent_submission_agent).launch()