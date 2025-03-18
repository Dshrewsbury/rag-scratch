# RAG-scratch


This repository contains a Python program that
exposes a RAG pipeline via FastAPI. It is designed to serve as a reference implementation
for AI applications. A lot of things such as recursive chunking or naive semantic search are implemented and are clearly not the ideal way of doing things. It's meant to be a basic reference for those starting off with RAG. So many Medium articles are just thrown together Langchain tutorials that provide no actual benefit towards learning of actual RAG concepts.

The project is implemented without the use of any LLM framework. The best Frameworks in this field are changing rapidly and many(I'm looking at you Langchain) have quickly gone out of control with countless abstractions and poor documentation. It's often best to fit your chunking, retrieval, and generation to your specific problem and frameworks all to often get in the way of that. Plus this is a great way to learn something. Once you've done it from scratch it's no problem to quickly pick up a framework if needed.

It includes the following:

- 🏎️ **FastAPI** – A type-safe, asynchronous web framework for building REST APIs.
- 🍓 **LiteLLM** – A proxy to call 100+ LLM providers from the OpenAI library.
- 🔍 **Qdrant** – A vector database for semantic, keyword, and hybrid search.
- 🚚 **UV** – A project and dependency manager.
- 🧪 **Pytest** – A testing framework.
- 🏗 **Testcontainers** – A tool to set up integration tests.
- 🐳 **Docker** – A tool to containerize the Python application.
- 🐙 **Compose** – A container orchestration tool for managing the application infrastructure.

##  Prerequisites

There are only two requirements to run this project:

- UV ([install](https://docs.astral.sh/uv/getting-started/installation/))
- Docker ([install](https://docs.docker.com/get-docker/))


## Usage

Follow the README in rag->processing folder for information on embedding generation.


### Setting up the infrastructure

Before running the application, you need to set up the environment variables.

Make a copy of the `.env.example` file and name it as `.env`.
Put your API keys and other environment variables in your `.env` file.

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Dshrewsbury/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   docker-compose up --build

 Visit the app at:  http://localhost:8000


## Testing

Evaluations for a RAG system are still an ongoing area of research. Many proposed metrics require the use of an LLM judge, and well using an LLM to judge an LLM(scary) isn't particularly reliable. Oftentimes optimizing the retrieval portion of your pipeline is the way to go, via Recall@k and Precision@k. 

With RAG systems, its advisable to start with a simple setup, with evals/basic metrics in place so you can make data-driven development decisions. Unfortunately a lot of things such as chunking strategies still require a human-in-the-loop, and golden test data is desireable.

I've including some very, very basic unit, integration, and e2e tests and integrated them into a CI/CD pipeline via Github Actions. 

I've also included evaluations as well as a synthetic data generation method, but these are ongoing and will be updated shortly.

## Upcoming Work

Planned improvements include:

- Streamlined Evaluation Metrics + Observability
- Improved Chunking
- Actual decent automated tests connected to metrics
- Container/Testing/Deployment that more mimics a production-level scenario
- Evals for an in production system(Topic Modeling, etc)
- An actual datapipeline for a real scenario
- Better config setup(sorry)
- Just
- Typechecking
