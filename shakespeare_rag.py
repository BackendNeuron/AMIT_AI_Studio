from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # or: from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # or: from langchain_ollama import OllamaLLM

class ShakespeareStyler:
    def __init__(self, vector_index_path="vector_index", model_name="qwen2.5:7b"):
        # Embeddings and vector DB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectordb = FAISS.load_local(
            vector_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # LLM (streaming enabled)
        self.llm = Ollama(model=model_name)

        # Prompt template
        self.prompt_template = PromptTemplate(
            template="""
You are a Shakespearean text stylist.
Take the user's input text and rewrite it in Shakespearean style, using the retrieved context from Shakespeare's sonnets for inspiration.

⚠️ Rules (STRICT):

The output must contain the exact same number of sentences as the input. No more, no less.

Each sentence must stay approximately the same length as the corresponding input sentence. Do not expand short input into long poetic lines.

Do not introduce new ideas, imagery, or themes that are not already in the input.

Only change the wording and tone into Shakespearean/Elizabethan style.

If the input is short, the output must remain short. If the input is long, the output may be long, but always in balance with the input.

Output must strictly preserve the meaning of the input.

User Input: {query}

Context (from sonnets):
{context}

Shakespearean Style Output:
""",
            input_variables=["query", "context"]
        )

    def style_text_stream(self, user_text: str):
        """
        Generator function: yields Shakespearean text letter by letter
        """
        retriever = self.vectordb.as_retriever()
        docs = retriever._get_relevant_documents(user_text, run_manager=None)  # fixed for current version

        context = "\n".join([doc.page_content for doc in docs])
        prompt_text = self.prompt_template.format(query=user_text, context=context)

        for token in self.llm.stream(prompt_text):
            for char in token:  # split token into letters
                yield char
