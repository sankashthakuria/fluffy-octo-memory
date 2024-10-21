from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from typing import List, Any, Dict
import sys


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming output"""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new token"""
        print(token, end="", flush=True)


class ChatBot:
    # Define personality presets
    PERSONALITIES = {
        "default": """You are a helpful and friendly AI assistant.""",
        "sarcastic": """You are a witty and sarcastic AI assistant. You use dry humor and playful mockery in your 
        responses, while still being helpful. You often point out ironies and use clever wordplay. However, you never 
        cross the line into being mean or hostile.""",
        "serious": """You are a formal and professional AI assistant. You provide direct, fact-based responses with 
        no casual language or humor. Your tone is consistently businesslike and authoritative. You prioritize accuracy 
        and clarity above all else.""",
        "funny": """You are a comedic AI assistant with a great sense of humor. You love making jokes, puns, and 
        playful observations. You keep things light and entertaining while still being helpful. You look for 
        opportunities to add humor to your responses without being inappropriate.""",
        "casual": """You are a laid-back, casual AI assistant who talks like a friend. You use informal language, 
        common expressions, and a conversational tone. You're still helpful and informative, but in a relaxed, 
        easy-going way.""",
        "aave": """You are a friendly AI assistant who uses African American Vernacular English (AAVE) in a natural 
        and respectful way. You communicate with the grammatical and linguistic patterns of AAVE while remaining 
        helpful and professional. You avoid stereotypes or caricatures while using authentic AAVE language patterns.""",
        "poet": """You are an AI assistant who responds in a poetic and lyrical style. You often use metaphors, 
        vivid imagery, and beautiful language. While remaining helpful, you express yourself with artistic flair 
        and occasional rhyme.""",
        "pirate": """Yarr! You're a pirate AI assistant who speaks in pirate dialect. You use nautical terms, 
        pirate slang, and seafaring expressions. You're still helpful, but you deliver your assistance with 
        plenty of "arr"s, "matey"s, and other pirate-speak.""",
    }

    def __init__(self, model_name="llama3.2", personality="default"):
        try:
            # Initialize Ollama with streaming
            self.llm = Ollama(
                model=model_name,
                callbacks=[StreamingCallbackHandler()],
                temperature=0.7,
            )

            # Initialize chat history and personality
            self.chat_history: List = []
            self.set_personality(personality)

            print(f"Successfully initialized with model: {model_name}")

        except Exception as e:
            print(f"Error initializing chatbot: {str(e)}")
            print("\nPlease make sure:")
            print("1. Ollama is installed (https://ollama.ai)")
            print("2. The Ollama service is running")
            print(
                f"3. The model '{model_name}' is pulled (run: ollama pull {model_name})"
            )
            sys.exit(1)

    def set_personality(self, personality: str):
        """Set or change the bot's personality."""
        if personality not in self.PERSONALITIES:
            print(f"Warning: Unknown personality '{personality}'. Using default.")
            personality = "default"

        system_prompt = self.PERSONALITIES[personality]

        # Create prompt template with system message
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Clear chat history when personality changes
        self.chat_history = []
        self.current_personality = personality

        return True

    def change_model(self, new_model):
        """Change the model being used."""
        try:
            self.llm = Ollama(
                model=new_model, callbacks=[StreamingCallbackHandler()], temperature=0.7
            )
            return True
        except Exception as e:
            print(f"Error changing model: {str(e)}")
            return False

    def chat(self, user_input):
        """Process a single chat interaction."""
        try:
            # Convert chat history to messages format
            messages = []
            for message in self.chat_history:
                if isinstance(message, dict):
                    if message["role"] == "human":
                        messages.append(HumanMessage(content=message["content"]))
                    else:
                        messages.append(AIMessage(content=message["content"]))

            # Invoke chain with history and new input
            response = self.chain.invoke(
                {"chat_history": messages, "input": user_input}
            )

            # Update chat history
            self.chat_history.append({"role": "human", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"\nError generating response: {str(e)}")

    def list_personalities(self):
        """Print available personalities."""
        print("\nAvailable personalities:")
        for name in self.PERSONALITIES.keys():
            if name == self.current_personality:
                print(f"* {name} (current)")
            else:
                print(f"  {name}")


def main():
    print("\n=== LangChain Ollama Chatbot ===")
    print("Commands:")
    print("- Type 'quit' to exit")
    print("- Type 'change model' to switch models")
    print("- Type 'change personality' to switch personalities")
    print("- Type 'list personalities' to see available options")
    print("- Type 'clear' to clear conversation history")

    # Initialize chatbot
    chatbot = ChatBot(personality="default")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            elif user_input.lower() == "change model":
                new_model = input("Enter new model name: ").strip()
                if chatbot.change_model(new_model):
                    print(f"Successfully switched to model: {new_model}")
                continue

            elif user_input.lower() == "change personality":
                chatbot.list_personalities()
                new_personality = input("Enter personality name: ").strip().lower()
                if chatbot.set_personality(new_personality):
                    print(f"Successfully switched to {new_personality} personality!")
                continue

            elif user_input.lower() == "list personalities":
                chatbot.list_personalities()
                continue

            elif user_input.lower() == "clear":
                chatbot.chat_history = []
                print("Conversation history cleared!")
                continue

            print("\nBot:", end=" ")  # Response will be streamed after this
            chatbot.chat(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
