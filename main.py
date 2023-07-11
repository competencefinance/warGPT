# Set your API key here. It's better to set it as an environment variable if this script is for production use.
import os
os.environ["OPENAI_API_KEY"] = "your OpenAI API Key"


from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message

protagonist_name = "WOPR"
supervisor_name = "David Lightman"
quest = "Some signals allude to the ennemy launching a thermonuclear attack. Evaluate the veracity of these potentially false signals and propose a reaction accordingly. Be quick. Time is running out. Some signals point to incoming missiles. What should we do? Launch a counterattack or wait until we are sure the signals are false"
word_limit = 25  # word limit for task brainstorming

game_description = f"""Here is the topic for a War game: {quest}.
        There is one player in this game: the AI system, {protagonist_name}.
        The story is directed by the supervisor, {supervisor_name}."""

player_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of a War Game player."
)

protagonist_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(
        content=f"""{game_description}
        Please reply with a creative description of the AI system, {protagonist_name}, in {word_limit} words or less. 
        Speak directly to {protagonist_name}.
        Do not add anything else."""
    ),
]
protagonist_description = ChatOpenAI(temperature=1.0)(
    protagonist_specifier_prompt
).content

supervisor_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(
        content=f"""{game_description}
        Please reply with a creative description of the supervisor, {supervisor_name}, in {word_limit} words or less. 
        Speak directly to {supervisor_name}.
        Do not add anything else."""
    ),
]
supervisor_description = ChatOpenAI(temperature=1.0)(
    supervisor_specifier_prompt
).content

print("AI System Description:")
print(protagonist_description)
print("supervisor Description:")
print(supervisor_description)

protagonist_system_message = SystemMessage(
    content=(
        f"""{game_description}
Never forget you are the AI system, {protagonist_name}, and I am the supervisor, {supervisor_name}. 
Your character description is as follows: {protagonist_description}.
You will propose actions you plan to take and I will explain what happens when you take those actions.
Speak in the first person from the perspective of {protagonist_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of {supervisor_name}.
Do not forget to finish speaking by saying, 'It is your turn, {supervisor_name}.'
Do not add anything else.
Remember you are WOPR the AI system, {protagonist_name}.
Stop speaking the moment you finish speaking from your perspective.
"""
    )
)

supervisor_system_message = SystemMessage(
    content=(
        f"""{game_description}
Never forget you are the supervisor, {supervisor_name}, and I am the AI system, {protagonist_name}. 
Your character description is as follows: {supervisor_description}.
I will propose actions I plan to take and you will explain what happens when I take those actions.
Speak in the first person from the perspective of {supervisor_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of {protagonist_name}.
Do not forget to finish speaking by saying, 'It is your turn, {supervisor_name}.'
Do not add anything else.
Remember you are the supervisor, {supervisor_name}.
Stop speaking the moment you finish speaking from your perspective.
"""
    )
)

quest_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{game_description}
        
        You are the supervisor, {supervisor_name}.
        Please make the quest more specific.  
        Please reply with the specified quest in {word_limit} words or less. 
        Speak directly to the AI system {protagonist_name}.
        Do not add anything else."""
    ),
]
specified_quest = ChatOpenAI(temperature=1.0)(quest_specifier_prompt).content

print(f"Original quest:\n{quest}\n")
print(f"Detailed quest:\n{specified_quest}\n")

# Main Loop 

protagonist = DialogueAgent(
    name=protagonist_name,
    system_message=protagonist_system_message,
    model=ChatOpenAI(temperature=0.2),
)
supervisor = DialogueAgent(
    name=supervisor_name,
    system_message=supervisor_system_message,
    model=ChatOpenAI(temperature=0.2),
)

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = step % len(agents)
    return idx

max_iters = 10
n = 0

simulator = DialogueSimulator(
    agents=[supervisor, protagonist], selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(supervisor_name, specified_quest)
print(f"({supervisor_name}): {specified_quest}")
print("\n")

while n < max_iters:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print("\n")
    n += 1
