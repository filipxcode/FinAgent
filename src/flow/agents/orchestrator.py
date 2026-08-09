from dataclasses import dataclass
from inspect import cleandoc

from pydantic_ai import RunContext

from src.config.config import get_settings
from src.flow.types import (
    BasicMessage,
    LanguageEnum,
    NodeABC,
    NodeContext,
    NodeInput,
    NodeOutput,
    NodeRunResult,
)


class OrchestratorInput(NodeInput):
    language: LanguageEnum
    

class OrchestratorOutput(NodeOutput):
    pass


class OrchestratorContext(NodeContext):
    pass


@dataclass
class OrchestratorDeps:
    messages: list[BasicMessage]

ORCHESTRATOR_AGENT_KEY = "orchestrator"
agent = get_settings().get_agent(ORCHESTRATOR_AGENT_KEY, deps_type=OrchestratorDeps,
    output_type=OrchestratorOutput,
)

@agent.instructions
async def get_agent_instructions(ctx: RunContext[OrchestratorDeps]) -> str:
    prompt ="""
    ###Input
    Last user message: 
    '''python
    {last_user_message}
    '''
    Message history
    '''python
    {messages}
    '''
    """
    prompt = cleandoc(prompt)
    last_user_message = ctx.deps.messages[-1].content if ctx.deps.messages else ""
    return prompt.format(
        last_user_message=last_user_message,
        messages=[m.model_dump() for m in ctx.deps.messages[:-1]],
    )


@dataclass(kw_only=True)
class OrchestratorNode(NodeABC[OrchestratorInput, OrchestratorContext, OrchestratorOutput]):
    name: str = "orchestrator_node"

    async def run_node(
        self,
        input: OrchestratorInput,
        context: OrchestratorContext,
    ) -> NodeRunResult[NodeOutput]:
        return NodeRunResult(
            output=NodeOutput(output_messages=[], next_node=None),
            analytics_params={"node": self.name, "cost": 0.0},
        )
