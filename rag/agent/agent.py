import asyncio
import logging
import traceback
import json
from pathlib import Path
import litellm
from litellm import acompletion
from litellm.utils import trim_messages
from litellm.caching.caching import Cache
from typing import Dict, Any, AsyncIterator
from rag.retrieval.vector_store import VectorStore
from rag.processing.embedding.embedding_generator import EmbeddingGenerator
from rag.memory.memory_database import MemoryDatabase
from traceloop.sdk import Traceloop  # type: ignore
from traceloop.sdk.decorators import workflow  # type: ignore


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
agent_dir = Path(__file__).parent
system_prompt_txt = agent_dir / "system_prompt.txt"
tools_json = agent_dir / "tools.json"

litellm.cache = Cache(type="redis", host="redis", port="6379", password="myredissecret")  # type: ignore
Traceloop.init(disable_batch=True)  # Disable when running locally


class Agent:
    def __init__(self, model_config=None):
        """
        Agent encapsulates a system prompt, tool definitions, and tool execution logic

        Args:
            model_config: Configuration for the model (defaults to OpenAI gpt-4o)
        """
        self.system_prompt = (
            system_prompt_txt.read_text() if system_prompt_txt.exists() else ""
        )
        self.tools = json.loads(tools_json.read_text()) if tools_json.exists() else []

        # Set default model config if not provided
        self.model_config = model_config or {
            "model": "gpt-4o-mini",
        }

        # Initialize components
        self.vector_store = VectorStore()
        self.embedding_generator = EmbeddingGenerator()
        self.memory_db = MemoryDatabase()
        print(self.memory_db.db_path)

        # Define tool mapping
        self.tool_map = {
            "search": self._search,
            "analyze_document": self._analyze_document,
            "improve_query": self._improve_query,
            "answer_factual": self._answer_factual,
        }

        # Track conversations
        self.conversations = {}

        # Log initialization
        logger.info(f"Agent initialized with model: {self.model_config.get('model')}")

    @workflow(name="general_response")
    async def generate_response_stream(
        self, conversation_id: str, query: str
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response for the conversation using LiteLLM's async streaming.

        Args:
            conversation_id: ID of the conversation
            query: User query

        Yields:
            Tokens as they are generated
        """
        # Initialize or get conversation history
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # Add user query to conversation history
        self.conversations[conversation_id].append({"role": "user", "content": query})

        # Prepare messages with system prompt and conversation history
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversations[conversation_id],
        ]

        # Initialize assistant message for storing the complete response
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": ""}

        try:
            # Log request
            logger.info(
                f"Generating streaming response with model: {self.model_config.get('model')}"
            )
            model = self.model_config["model"]

            # Get the response stream
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

            # Iterate through streaming chunks asynchronously
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
                    # Store tool calls in assistant message
                    if "tool_calls" not in assistant_message:
                        assistant_message["tool_calls"] = []

                    # Process each tool call
                    for tool_call in tool_calls:
                        if (
                            not tool_call.get("id")
                            or tool_call["id"] in tool_calls_seen
                        ):
                            continue

                        tool_calls_seen.append(tool_call["id"])
                        assistant_message["tool_calls"].append(tool_call)

                        # Extract tool information
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])

                        # Notify user that a tool is being used
                        yield f"\n\n_Using {tool_name} tool..._\n"

                        # Execute tool and get result
                        tool_result = await self.execute_tool(
                            tool_name=tool_name, **tool_args
                        )

                        # Add tool result to messages
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

                        # Continue the conversation with tool results
                        yield f"\n_Tool result:_\n{tool_result}\n\n"

                        # Start a new completion with the tool results
                        additional_response = await acompletion(
                            model=self.model_config["model"],
                            messages=messages,
                            stream=True,
                            tools=self.tools,
                        )

                        # Stream the additional response
                        async for add_chunk in additional_response:
                            if (delta := add_chunk.choices[0].delta) and (
                                token := delta.content
                            ):
                                assistant_message["content"] += token
                                yield token

            # Update conversation history with assistant's response
            self.conversations[conversation_id].append(assistant_message)

        except Exception as e:
            # Log the full traceback for debugging
            error_traceback = traceback.format_exc()
            logger.error(f"Error during streaming: {str(e)}\n{error_traceback}")

            # Yield error information to the stream
            error_msg = f"Error during streaming: {str(e)}"
            yield f"\n\n_Error: {error_msg}_"

    async def generate_response(self, conversation_id: str, query: str) -> str:
        """
        Generate a complete non-streaming response.

        Args:
            conversation_id: ID of the conversation
            query: User query

        Returns:
            Complete generated response
        """
        # Initialize or get conversation history
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # Add user query to conversation history
        self.conversations[conversation_id].append({"role": "user", "content": query})

        # Prepare messages with system prompt and conversation history
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversations[conversation_id],
        ]

        try:
            # Log request parameters for debugging
            logger.info(
                f"Generating complete response with model: {self.model_config.get('model')}"
            )

            # Generate complete response
            response = await acompletion(
                model=self.model_config["model"],
                messages=messages,
                # tools=self.tools
            )

            # Extract and process the response
            response_message = response.choices[0].message
            response_text = response_message.content or ""

            # Check for tool calls
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                tool_results = []

                # Process each tool call
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Execute tool
                    tool_result = await self.execute_tool(
                        tool_name=tool_name, **tool_args
                    )

                    # Add results to list
                    tool_results.append({"tool": tool_name, "result": tool_result})

                    # Add to conversation history
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

                # Use the updated messages to generate final response
                final_response = await acompletion(
                    model=self.model_config["model"], messages=messages
                )

                # Get the final text
                response_text = final_response.choices[0].message.content

                # Add summary of tools used
                tools_summary = "\n\n_Tools used:_\n"
                for result in tool_results:
                    tools_summary += f"- {result['tool']}\n"

                response_text = response_text + tools_summary

            # Add to conversation history
            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response_text}
            )

            return response_text

        except Exception as e:
            # Log the full traceback for debugging
            error_traceback = traceback.format_exc()
            logger.error(f"Error generating response: {str(e)}\n{error_traceback}")

            # Return error message with debug info
            response_text = f"Error generating response: {str(e)}"

            return response_text

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments for the tool

        Returns:
            Result of the tool execution
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
        Search for information in the knowledge base.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            Search results as text
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

            # Extract and return context
            if search_results:
                results_text = ""
                for i, row in enumerate(search_results):
                    results_text += f"Result {i + 1}:\n{row.payload['text']}\n\n"

                logger.info(f"Retrieved {len(search_results)} search results")
                return results_text

            logger.info("No search results found")
            return "No results found matching your query"

        except Exception as e:
            # Log error but don't fail
            logger.error(f"Error in search: {str(e)}")
            return f"Search error: {str(e)}"

    async def _analyze_document(self, text: str) -> str:
        """
        Extract key information from a document including main topics, entities, and summary.

        Args:
            text: Text to analyze

        Returns:
            Analysis results
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

        Args:
            query: Original query

        Returns:
            Improved query
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
        Answer factual questions without needing to search when confident.

        Args:
            question: Factual question

        Returns:
            Answer or indication that search is needed
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
        """Generate a title for a conversation based on the first message."""
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
