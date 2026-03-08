---
name: langchain-rag-pipeline
description: Build RAG pipelines with LangChain, FAISS vector stores, and LLMs (Ollama or OpenAI). Use when creating retrieval-augmented generation systems, building vector stores from data, setting up LangChain chains, adding conversation memory, or evaluating LLM responses.
---

# LangChain RAG Pipeline

## Quick Start — Minimal RAG Chain

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2:3b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

docs = [Document(page_content="...", metadata={"type": "overview"})]
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("query")
context = "\n".join([d.page_content for d in results])

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided context."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": "query"})
```

## Document Creation from DataFrames

Convert aggregated DataFrame stats into LangChain Documents for retrieval:

```python
documents = []

# Overview document
documents.append(Document(
    page_content=f"Dataset: {len(df)} rows. Total Sales: ${df['Sales'].sum():,.2f}. ...",
    metadata={"type": "overview"},
))

# Per-category documents
for cat in df["Category"].unique():
    subset = df[df["Category"] == cat]
    documents.append(Document(
        page_content=f"{cat}: ${subset['Sales'].sum():,.2f}, {len(subset)} txns, ...",
        metadata={"type": "category"},
    ))

# Cross-dimensional documents
for cat in df["Category"].unique():
    for region in df["Region"].unique():
        subset = df[(df["Category"] == cat) & (df["Region"] == region)]
        if len(subset) > 0:
            documents.append(Document(
                page_content=f"{cat} in {region}: ${subset['Sales'].sum():,.2f}, ...",
                metadata={"type": "cross"},
            ))
```

Pack each document with pre-computed stats (totals, averages, medians, counts, percentages) so the LLM can answer without calculation.

## LLM Providers

### Ollama (local, free)
```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
llm = ChatOllama(model="llama3.2:3b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### OpenAI (cloud, paid)
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()
```

### HuggingFace Embeddings (local, free)
```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

## Prompt Templates

### Simple Q&A
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a BI analyst. Answer using ONLY the data context provided. Be precise with numbers."),
    ("human", "Data Context:\n{context}\n\nQuestion: {question}"),
])
```

### Structured Output
```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

schemas = [
    ResponseSchema(name="summary", description="Brief answer"),
    ResponseSchema(name="details", description="Supporting data points"),
    ResponseSchema(name="recommendation", description="Actionable suggestion"),
]
parser = StructuredOutputParser.from_response_schemas(schemas)
```

## Memory Integration

### Buffer Memory (stores full history)
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
```

### Summary Memory (compresses history)
```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
```

### Conversational Retrieval Chain
```python
from langchain.chains import ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, return_source_documents=True,
)
result = chain({"question": "query"})
```

## Evaluation with QAEvalChain

```python
from langchain.evaluation.qa import QAEvalChain

examples = [
    {"query": "Which product has highest sales?", "answer": "Widget A"},
]

predictions = [{"result": chain.run(ex["query"])} for ex in examples]
eval_chain = QAEvalChain.from_llm(llm)
grades = eval_chain.evaluate(examples, predictions)
```

## FAISS Operations

```python
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")
loaded = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore.add_documents([new_doc])
results = vectorstore.similarity_search("query", k=5)
```
