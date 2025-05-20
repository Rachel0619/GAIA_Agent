from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "answer_question"
    description = "Analyzes a question and provides a detailed answer."
    inputs = {'question': {'type': 'string', 'description': 'The question to be answered'}}
    output_type = "string"

    def forward(self, question: str) -> str:
        """
        Analyzes a question and provides a detailed answer.

        Args:
            question (str): The question to be answered

        Returns:
            str: A comprehensive answer to the question
        """
        # Your logic here
        return "Your answer"