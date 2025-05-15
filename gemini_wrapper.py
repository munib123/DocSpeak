import google.generativeai as genai

class GoogleGeminiWrapper:
    def __init__(self, api_key: str):
        """
        Initialize the GoogleGeminiWrapper with the API key.

        :param api_key: Your Google Gemini API key.
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.conversation_history = [] # For the new chat method
        self.chat_session = None # To store the chat session for gemini

    def ask(self, prompt: str, model: str = "gemini-2.0-flash", max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        Send a prompt to the Google Gemini model and get a response (single turn).

        :param prompt: The input prompt to send to the model.
        :param model: The model to use (default is "gemini-pro").
        :param max_tokens: The maximum number of tokens to include in the response.
        :param temperature: Sampling temperature (higher values mean more randomness).
        :return: The response from the model as a string.
        """
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            model_instance = genai.GenerativeModel(model_name=model, generation_config=generation_config)
            response = model_instance.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"An error occurred: {e}"

    def start_chat_session(self, model: str = "gemini-2.0-flash", temperature: float = 0.7, max_tokens: int = 150):
        """
        Starts a new chat session or continues an existing one.
        """
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        model_instance = genai.GenerativeModel(model_name=model, generation_config=generation_config)
        # For Gemini, conversation history is managed by the chat object itself.
        # We re-initialize the chat session if one doesn't exist or if we want to start fresh.
        # If you want to persist history across calls to `chat` without explicitly calling reset,
        # you might initialize `self.chat_session` in `__init__` or when `chat` is first called.
        self.chat_session = model_instance.start_chat(history=self.conversation_history)


    def chat(self, prompt: str, model: str = "gemini-2.0-flash", max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        Send a prompt to the Google Gemini model, maintaining conversation history for context.

        :param prompt: The input prompt to send to the model.
        :param model: The model to use (default is "gemini-pro").
        :param max_tokens: The maximum number of tokens to include in the response.
        :param temperature: Sampling temperature (higher values mean more randomness).
        :return: The response from the model as a string.
        """
        try:
            if self.chat_session is None:
                self.start_chat_session(model=model, temperature=temperature, max_tokens=max_tokens)
            
            response = self.chat_session.send_message(prompt)
            assistant_response = response.text.strip()
            
            # Gemini's chat session object updates its history internally.
            # We can optionally also store it in our self.conversation_history if needed for other purposes
            # or if we want to be able to reconstruct the chat session later.
            # For simplicity here, we rely on the chat_session's internal history.
            # To manually track:
            # self.conversation_history.append({"role": "user", "parts": [prompt]})
            # self.conversation_history.append({"role": "model", "parts": [assistant_response]})
            
            return assistant_response
        except Exception as e:
            # Reset chat session on error to avoid issues with subsequent calls
            self.chat_session = None 
            return f"An error occurred: {e}"

    def reset_conversation(self):
        """
        Reset the conversation history and the chat session.
        """
        self.conversation_history = []
        self.chat_session = None # Crucial for Gemini to start a fresh chat

    def list_available_models(self):
        """
        Lists available Gemini models.
        :return: A list of available models.
        """
        try:
            print("Available Gemini Models:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(m.name)
            return [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except Exception as e:
            return f"An error occurred while listing models: {e}"

# Example usage (uncomment to test):
if __name__ == "__main__":
    api_key = "AIzaSyBisxoehBz8UF0i9kX42f1V3jp-9RNq04g" # Replace with your actual key
    wrapper = GoogleGeminiWrapper(api_key)

    # Example 0: List available models
    # print("\nListing available models...")
    # available_models = wrapper.list_available_models()
    # The function already prints, but you can use the returned list if needed
    # print(available_models) 
#
    # Example 1: Simple one-off question
    response_ask = wrapper.ask("What is the largest planet in our solar system?")
    print(f"Ask response: {response_ask}")
#
#     # Example 2: Conversation with history
#     print("\nStarting chat conversation...")
#     response1 = wrapper.chat("Hi, my name is Alex.")
#     print(f"Chat response 1: {response1}")
#
#     response2 = wrapper.chat("What is my name?")
#     print(f"Chat response 2: {response2}") # Should remember "Alex"
#
#     response3 = wrapper.chat("What was the first thing I asked you in this chat?")
#     print(f"Chat response 3: {response3}")
#
#     # Reset conversation history
#     wrapper.reset_conversation()
#     print("\nConversation reset.")
#
#     response4 = wrapper.chat("Do you remember my name?")
#     print(f"Chat response 4 (after reset): {response4}") # Should not remember "Alex"
