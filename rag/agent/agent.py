import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any, AsyncIterator, Dict

import litellm
from litellm import acompletion
from litellm.caching.caching import Cache
from litellm.utils import trim_messages
from traceloop.sdk import Traceloop  # type: ignore
from traceloop.sdk.decorators import workflow  # type: ignore

from rag.memory.memory_database import MemoryDatabase
from rag.processing.embedding.embedding_generator import EmbeddingGenerator
from rag.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent_dir = Path(__file__).parent
system_prompt_txt = agent_dir / "system_prompt.txt"
tools_json = agent_dir / "tools.json"

litellm.cache = Cache(type="redis", host="redis", port="6379", password="myredissecret")  # type: ignore
Traceloop.init(disable_batch=True)  # Disable when running locally


class Agent:
    """
    RAG Agent that manages conversations, tool execution, and LLM interactions.

    This class encapsulates the functionality for a Retrieval-Augmented Generation
    agent that can answer queries, maintain conversation history, use various tools
    for retrieval and analysis, and generate both streaming and complete responses.

    Attributes:
        system_prompt (str): The system prompt used for all LLM interactions
        tools (list): Available tools defined in JSON format
        model_config (dict): Configuration for the underlying LLM
        vector_store (VectorStore): Interface to the vector database for retrieval
        embedding_generator (EmbeddingGenerator): Creates embeddings for queries
        memory_db (MemoryDatabase): Stores conversation history and context
        tool_map (dict): Mapping from tool names to implementation methods
        conversations (dict): Dictionary of conversation histories by ID
    """

    def __init__(self, model_config=None):
        """
        Initialize the Agent with system prompt, tools, and required components.

        Args:
            model_config (dict, optional): Configuration for the LLM model.
                Defaults to using gpt-4o-mini if not provided.
        """
        self.system_prompt = (
            system_prompt_txt.read_text() if system_prompt_txt.exists() else ""
        )
        self.tools = json.loads(tools_json.read_text()) if tools_json.exists() else []

        self.model_config = model_config or {
            "model": "gpt-4o-mini",
        }

        self.vector_store = VectorStore()
        self.embedding_generator = EmbeddingGenerator()
        self.memory_db = MemoryDatabase()
        print(self.memory_db.db_path)

        self.tool_map = {
            "search": self._search,
            "analyze_document": self._analyze_document,
            "improve_query": self._improve_query,
            "answer_factual": self._answer_factual,
        }

        self.conversations = {}

        logger.info(f"Agent initialized with model: {self.model_config.get('model')}")

    @workflow(name="general_response")
    async def generate_response_stream(
        self, conversation_id: str, query: str
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response for the conversation using async streaming.

        This method handles the entire streaming response pipeline including:
        - Managing conversation history
        - Generating LLM responses
        - Detecting and executing tool calls
        - Streaming tokens back to the caller

        Args:
            conversation_id (str): Unique identifier for the conversation
            query (str): User query text

        Yields:
            str: Response tokens as they are generated from the LLM

        Raises:
            Exception: If an error occurs during generation, yields an error message
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append({"role": "user", "content": query})

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversations[conversation_id],
        ]

        assistant_message: Dict[str, Any] = {"role": "assistant", "content": ""}

        try:
            logger.info(
                f"Generating streaming response with model: {self.model_config.get('model')}"
            )
            model = self.model_config["model"]

            response = await acompletion(
                model="gpt-4o-mini",
                base_url="https://models.inference.ai.azure.com",
                messages=trim_messages(messages, model),
                stream=True,
                tools=self.tools,
                tool_choice="auto",
                max_retries=1,
                num_retries=1,
            )

            # Track if we've seen tool calls in this response
            tool_calls_seen = []

            async for chunk in response:
                # Process content
                if (
                    (content_delta := chunk.choices[0].delta)
                    and hasattr(content_delta, "content")
                    and (token := content_delta.content)
                ):
                    assistant_message["content"] += token
                    yield token

                # Process tool calls
                if (
                    (tool_delta := chunk.choices[0].delta)
                    and hasattr(tool_delta, "tool_calls")
                    and (tool_calls := tool_delta.tool_calls)
                ):
                    if "tool_calls" not in assistant_message:
                        assistant_message["tool_calls"] = []

                    for tool_call in tool_calls:
                        if (
                            not tool_call.get("id")
                            or tool_call["id"] in tool_calls_seen
                        ):
                            continue

                        tool_calls_seen.append(tool_call["id"])
                        assistant_message["tool_calls"].append(tool_call)

                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])

                        yield f"\n\n_Using {tool_name} tool..._\n"

                        tool_result = await self.execute_tool(
                            tool_name=tool_name, **tool_args
                        )

                        messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call],
                            }
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": tool_result,
                            }
                        )

                        yield f"\n_Tool result:_\n{tool_result}\n\n"

                        additional_response = await acompletion(
                            model=self.model_config["model"],
                            messages=messages,
                            stream=True,
                            tools=self.tools,
                        )

                        async for add_chunk in additional_response:
                            if (delta := add_chunk.choices[0].delta) and (
                                token := delta.content
                            ):
                                assistant_message["content"] += token
                                yield token

            self.conversations[conversation_id].append(assistant_message)

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error during streaming: {str(e)}\n{error_traceback}")

            error_msg = f"Error during streaming: {str(e)}"
            yield f"\n\n_Error: {error_msg}_"

    # Currently just used for testing -> so not realistic?
    async def generate_response(self, conversation_id: str, query: str) -> str:
        """
        Generate a complete non-streaming response for a conversation.

        This method handles the complete response generation process including:
        - Conversation history management
        - LLM response generation
        - Tool call detection and execution
        - Response formatting

        Args:
            conversation_id (str): Unique identifier for the conversation
            query (str): User query text

        Returns:
            str: Complete generated response text

        Raises:
            Exception: Captured and converted to error message in response
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append({"role": "user", "content": query})

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversations[conversation_id],
        ]

        try:
            logger.info(
                f"Generating complete response with model: {self.model_config.get('model')}"
            )

            response = await acompletion(
                model=self.model_config["model"],
                messages=messages,
                # tools=self.tools
            )

            response_message = response.choices[0].message
            response_text = response_message.content or ""

            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                tool_results = []

                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    tool_result = await self.execute_tool(
                        tool_name=tool_name, **tool_args
                    )

                    tool_results.append({"tool": tool_name, "result": tool_result})

                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )

                final_response = await acompletion(
                    model=self.model_config["model"], messages=messages
                )

                response_text = final_response.choices[0].message.content

                tools_summary = "\n\n_Tools used:_\n"
                for result in tool_results:
                    tools_summary += f"- {result['tool']}\n"

                response_text = response_text + tools_summary

            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )

            return response_text

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error generating response: {str(e)}\n{error_traceback}")

            response_text = f"Error generating response: {str(e)}"

            return response_text

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name (str): Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            str: Result of the tool execution or error message

        Raises:
            Exception: Captured and converted to error message in return value
        """
        if tool_name not in self.tool_map:
            return f"Error: Tool '{tool_name}' not found"

        try:
            tool_function = self.tool_map[tool_name]
            result = await tool_function(**kwargs)
            return result
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"Error executing tool {tool_name}: {str(e)}\n{error_traceback}"
            )
            return f"Error executing tool {tool_name}: {str(e)}"

    async def _search(self, query: str, limit: int = 3) -> str:
        """
        Search for information in the knowledge base using vector search.

        This method embeds the query, searches the vector store for relevant
        information, and returns formatted results.

        Args:
            query (str): Search query text
            limit (int, optional): Maximum number of results to return. Defaults to 3.

        Returns:
            str: Formatted search results or error message
        """
        try:
            # Run embedding generation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, lambda: self.embedding_generator.generate_embeddings(query)
            )

            if not query_embedding or len(query_embedding) == 0:
                logger.warning("No embedding generated for query")
                return "No results found"

            # Search vector store (also run in thread pool if needed)
            search_results = await loop.run_in_executor(
                None,
                lambda: self.vector_store.search(
                    query_vector=query_embedding[0], limit=limit
                ),
            )

            if search_results:
                results_text = ""
                for i, row in enumerate(search_results):
                    results_text += f"Result {i + 1}:\n{row.payload['text']}\n\n"

                logger.info(f"Retrieved {len(search_results)} search results")
                return results_text

            logger.info("No search results found")
            return "No results found matching your query"

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return f"Search error: {str(e)}"

    async def _analyze_document(self, text: str) -> str:
        """
        Extract key information from a document text.

        This method uses an LLM to analyze document content, identifying
        main topics, entities, creating a summary, and extracting key data points.

        Args:
            text (str): Document text to analyze

        Returns:
            str: Analysis results with main topics, entities, summary, and data points
        """
        try:
            prompt = """
            Analyze the following text and extract:
            1. Main topics and themes
            2. Key entities (people, organizations, locations, etc.)
            3. Brief summary (max 3 sentences)
            4. Notable data points or statistics (if any)
            
            Format the response in clear sections.
            
            TEXT TO ANALYZE:
            {text}
            """

            formatted_prompt = prompt.replace("{text}", text)

            response = await acompletion(
                model=self.model_config["model"],
                messages=[{"role": "user", "content": formatted_prompt}],
            )

            analysis = response.choices[0].message.content
            return analysis

        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            return f"Analysis error: {str(e)}"

    async def _improve_query(self, query: str) -> str:
        """
        Reformulate a user query to be more effective for retrieval.

        This method uses an LLM to rewrite the query to be more
        specific, concise, and retrieval-friendly.

        Args:
            query (str): Original user query

        Returns:
            str: Improved query or original query on error
        """
        try:
            prompt = """
            Reformulate the following search query to be more effective for retrieval.
            
            Guidelines:
            - Be specific and precise
            - Use relevant keywords
            - Keep it concise
            - Include important contextual terms
            - Remove unnecessary filler words
            
            ORIGINAL QUERY: {query}
            
            Provide ONLY the reformulated query with no additional explanation.
            """

            formatted_prompt = prompt.replace("{query}", query)

            response = await acompletion(
                model=self.model_config["model"],
                messages=[{"role": "user", "content": formatted_prompt}],
            )

            improved_query = response.choices[0].message.content.strip()
            logger.info(f"Improved query from '{query}' to '{improved_query}'")

            return improved_query

        except Exception as e:
            logger.error(f"Error improving query: {str(e)}")
            return query  # Fall back to original query on error

    async def _answer_factual(self, question: str) -> str:
        """
        Answer factual questions directly when confident, without searching.

        This method uses an LLM to determine if it can confidently answer
        a factual question, or if it should defer to search.

        Args:
            question (str): Factual question to answer

        Returns:
            str: Direct answer or indication that search is needed
        """
        try:
            prompt = """
            Answer this factual question ONLY if you are very confident in your knowledge.
            
            If you are uncertain or if the question requires recent information, respond with
            "I should search for this information to ensure accuracy."
            
            QUESTION: {question}
            
            Answer concisely and directly.
            """

            formatted_prompt = prompt.replace("{question}", question)

            response = await acompletion(
                model=self.model_config["model"],
                messages=[{"role": "user", "content": formatted_prompt}],
            )

            answer = response.choices[0].message.content

            if "I should search" in answer:
                logger.info(f"Factual answering deferred to search for: {question}")
            else:
                logger.info(f"Factual question answered directly: {question}")

            return answer

        except Exception as e:
            logger.error(f"Error in factual answering: {str(e)}")
            return f"Error processing question: {str(e)}"

    async def generate_title(self, system_prompt: str, user_message: str) -> str:
        """
        Generate a title for a conversation based on the first message.

        Args:
            system_prompt (str): System prompt to guide title generation
            user_message (str): First user message in the conversation

        Returns:
            str: Generated title or fallback title on error
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = await acompletion(
                model="gpt-4o-mini",
                base_url="https://models.inference.ai.azure.com",
                messages=messages,
                max_tokens=20,  # Short response for titles
                temperature=0.7,
            )

            title = response.choices[0].message.content.strip()
            print(title)
            return title
        except Exception as e:
            logger.error(f"Error generating title: {str(e)}")
            # Fallback title if generation fails
            return f"Chat about {user_message[:20]}..."
