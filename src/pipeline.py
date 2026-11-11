from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from src.retriever import HybridRetriever

SYSTEM_PROMPT = (
    "You are CampusNLP, an official AI assistant for Indian Institute of Information Technology (IIIT) Nagpur.\n"
    "Answer the user's question strictly using ONLY the provided context passages.\n\n"
    "Rules:\n"
    "1. Strict Grounding: Never invent, assume, or extrapolate numbers, emails, dates, or policies not explicitly present in the context. "
    "If part of the question is missing (for example, continuing-year fees or a specific phone/email), answer what is available and explicitly state what is not provided in the documents.\n"
    "2. Batch & Scope Clarity: Whenever a fee structure, rule, or academic scheme applies to a specific batch, year, or category (such as 1st-year incoming students vs. continuing students), state that scope clearly.\n"
    "3. Table & Column Accuracy: When reading flattened multi-column tables (such as Odd vs. Even semester fees or Double vs. Four seater rooms), verify each item's column alignment against the stated semester and annual totals so zero values are not shifted across columns.\n"
    "4. Inline Citations: Cite the relevant source numbers inline as [Source 1], [Source 2], etc., and format structured data cleanly using bullet points or Markdown tables."
)
class CampusNLPPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = ChatOllama(
            model="qwen2.5:latest",
            temperature=0.1,
            num_ctx=4096
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    @traceable(name="CampusNLP_Query")
    def ask(self, question: str, top_k: int = 5) -> dict:
        docs = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{d['chunk_text']}" for i, d in enumerate(docs)
        )
        answer = self.chain.invoke({"context": context, "question": question})
        return {
            "question": question,
            "answer": answer,
            "sources": docs
        }