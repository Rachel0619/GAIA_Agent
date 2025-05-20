import yaml
import os
from smolagents import GradioUI, CodeAgent, InferenceClientModel

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.answer_question import SimpleTool as AnswerQuestion
from tools.web_search import DuckDuckGoSearchTool as WebSearch
from tools.visit_webpage import VisitWebpageTool as VisitWebpage
from tools.final_answer import FinalAnswerTool as FinalAnswer



model = InferenceClientModel(
model_name='Qwen/Qwen2.5-Coder-32B-Instruct',
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
)

answer_question = AnswerQuestion()
web_search = WebSearch()
visit_webpage = VisitWebpage()
final_answer = FinalAnswer()


with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent_submission_agent = CodeAgent(
    model=model,
    tools=[answer_question, web_search, visit_webpage],
    managed_agents=[],
    max_steps=5,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name='submission_agent',
    description='Agent for answering evaluation questions',
    executor_type='local',
    executor_kwargs={},
    max_print_outputs_length=None,
    prompt_templates=prompt_templates
)
if __name__ == "__main__":
    GradioUI(agent_submission_agent).launch()
