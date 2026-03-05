# 📈 Autonomous Multi-Agent Financial Analyst

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange.svg)](https://python.langchain.com/v0.1/docs/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

## Overview
An enterprise-grade, multi-agent AI pipeline designed to automate financial research and data visualization. By orchestrating specialized AI agents (Researcher, Coder, Critic) via **LangGraph**, this system autonomously ingests SEC 10-K filings, performs Retrieval-Augmented Generation (RAG), writes deterministic Python code for data visualization, and peer-reviews its own outputs to eliminate hallucinations.

## Architecture & Features
* **Stateful Multi-Agent Orchestration:** Utilizes LangGraph to manage a continuous state and multi-turn conversational memory (`SqliteSaver`).
* **Cost-Optimized LLM Routing:** Leverages **OpenRouter** to dynamically route simple extraction tasks to low-latency open-source models (Gemma-3), reserving heavy reasoning for code generation. Includes automatic fallback routing for API rate limits.
* **Zero-Cost Local RAG:** Implements a local vector database using **ChromaDB** and **HuggingFace** (`all-MiniLM-L6-v2`) embeddings to semantically search massive regulatory documents without incurring API token costs.
* **Autonomous Code Execution (PythonREPL):** A dedicated 'Data Scientist' agent writes, executes, and debugs `matplotlib` scripts in a sandboxed environment based on real SEC data.
* **Deterministic Infinite-Loop Protection:** A 'Senior Reviewer' agent grades outputs and forces revision loops (capped at 3 iterations) to ensure high-accuracy text and functional code.

## Live Demo
https://ai-financial-analysis-krcztgxet4pazpvklcavxm.streamlit.app/
