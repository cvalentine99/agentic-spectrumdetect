# agent.py
import os
import ssl
import httpx
import tiktoken
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from rich.live import Live
from rich.markdown import Markdown
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class MemoryManager:
    def __init__(self,
                 max_context_window: int = 131072,
                 target_context_usage: float = 0.9,
                 short_term_capacity: int = 5,
                 provider: Any = None,
                 model_name: str = None):
        self.short_term_memory: List[Dict[str, str]] = []
        self.long_term_memory: List[Dict[str, str]] = []
        self.max_context_window = max_context_window
        self.target_context_usage = target_context_usage
        self.target_token_count = int(max_context_window * target_context_usage)
        self.short_term_capacity = short_term_capacity
        self.provider = provider
        self.model_name = model_name
        self.summary: Optional[str] = None

        try:
            self.encdoing = tiktoken.get_encoding("o200k_base")
        except:
            self.encoding = tiktoken.encoding_for_model("gpt-oss-20b")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def count_message_tokens(self, message: Dict[str, str]) -> int:
        if not message:
            return 0

        content = message.get("content", "")
        role = message.get("role", "")

        return 4 + self.count_tokens(content) + len(role)

    def estimate_total_token_count(self, messages: List[Dict[str, str]]) -> int:
        if not messages:
            return 0
      
        return sum(self.count_message_tokens(msg) for msg in messages)

    async def add_to_memory(self, message: Dict[str, str]) -> None:
        self.short_term_memory.append(message)

        current_token_count = self.estimate_total_token_count(self.short_term_memory)

        if len(self.short_term_memory) > self.short_term_capacity or current_token_count > (self.target_token_count / 2):

            self.long_term_memory.append(self.short_term_memory.pop(0))
            if len(self.long_term_memory) >= 5:
                await self.summarize_long_term_memory()

    async def summarize_long_term_memory(self) -> None:
        if not self.provider or not self.model_name or not self.long_term_memory:
            return

        memory_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.long_term_memory
            ])

        summary_prompt = f"Please create a concise summary of the following conversation that captures the key points and context. This summary will be used to maintain context in future interactions.\n\nConversatoin:\n{memory_text}"

        model = OpenAIModel(
                model_name=self.model_name,
                provider=self.provider,
        )

        try:
            messages = [{"role": "user", "content": summary_promt}]
            result = await model.async_generate(messages=messages)

            updated_summary = result["choice"][0]["message"]["content"]
            if self.summary:
                self.summary = f"{self.summary}\n\nAdditional Context: {updated_summary}"
            else:
                self.summary = updated_summary

            summary_tokens = self.count_tokens(self.summary)
            if summary_tokens > (self.target_token_count * 0.3):
                condense_prompt = f"The following is a conversation summary that is getting to long. Please create a more concise version that preserves the key information:\n\n{self.summary}"
                messages = [{"role": "user", "content": condense_prompt}]
                result = await model.async_generate(messages=messages)
                self.summary = result["choices"][0]["message"]["content"]

            self.long_term_memory = []
        except Exception as e:
            print(f"Error during summarization: {e}")

    def get_messages_for_context(self) -> List[Dict[str, str]]:
        messages = []
        token_budget = self.target_token_count

        if self.summary:
            summary_message = {"role": "system", "content": f"Previous conversation context: {self.summary}"}
            summary_tokens - self.count_message_tokens(summary_message)

            if summary_tokens < token_budget:
                messages.append(summary_message)
                token_budget -= summary_tokens
            else:
                trucated_summary = self.summary[:int(len(self.summary) * (token_budget / summary_tokens) * 0.9)]
                truncated_message = {"role": "system", "content": f"Previous convesation (trucates): {truncated_summar}"}
                messages.append(trucates_message)
                token_budget -= self.count_message_tokens(truncated_message)

        for msg in reversed(self.short_term_memory):
            msg_tokens = self.count_message_tokens(msg)
            if msg_tokens < token_budget:
                messages.insert(1, msg)
                token_budget -= msg_tokens
            else:
                break

        if len(messages) > 1:
            start = 1 if messages[0].get("role") == "system" else 0
            messages[start:] = list(reversed(messages[start:]))

        return messages


class AgentRunner:
    def __init__(
        self,
        spectrum_server_mcp: str,
        llm_api: str,
        llm_api_key: str,
        llm_model: str,
        llm_reasoning: str,
        system_prompt: str,
        otel_exporter_otlp_endpoint: str,
        ca_chain: str | None = None,
        max_context_window: int = 131072,
    ):
        self.llm_key = llm_api_key
        self.mcp_server = spectrum_server_mcp
        self.llm_server_hostname = llm_api
        self.llm_model = llm_model
        self.llm_reasoning_level = llm_reasoning
        self.system_prompt = system_prompt
        self.max_context_window = max_context_window
        self.http_client: httpx.AsyncClient | None = None
        if ca_chain:
            self.ssl_ctx = ssl.create_default_context(cafile=ca_chain)
            self.http_client = httpx.AsyncClient(
                verify=self.ssl_ctx,
                timeout=httpx.Timeout(600.0),
            )
        if otel_exporter_otlp_endpoint:
            try:
                import logfire
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_exporter_otlp_endpoint
                logfire.configure(send_to_logfire=False)
                logfire.instrument_pydantic_ai()
                logfire.instrument_httpx(capture_all=True)
            except:
                pass

        self.console = Console()

    async def run(self):
        server = MCPServerStreamableHTTP(url=self.mcp_server)
        self.console.print("[white]Connecting to MCP server...[/white]")
        async with server:
            self.console.print(
                "[bold white]Connection successful. Initializing agent...[/bold white]"
            )

            local_provider = OpenAIProvider(
                base_url=self.llm_server_hostname,
                api_key=self.llm_key,
                http_client=self.http_client,
            )
            local_model = OpenAIModel(
                model_name=self.llm_model,
                provider=local_provider,
            )
            model_settings = OpenAIResponsesModelSettings(
                    openai_reasoning_effort=self.llm_reasoning_level,
                    openai_reasoning_summary='concise'
            )

            memory_manager = MemoryManager(
                    max_context_window=self.max_xontext_window,
                    target_context_usage=0.9,
                    short_term_capacity=15,
                    provider=local_provider,
                    model_name=self.llm_model
            )

            await memory_manager.add_to_memory({"role": "system", "content": self.system_prompt})

            agent = Agent(
                model=local_model, 
                model_settings=model_settings, 
                toolsets=[server], 
                system_prompt=self.system_prompt
            )

            self.console.print(
                "[bold green]Agent is ready. Type 'exit' or 'quit' to end.[/bold green]"
            )
            while True:
                try:
                    prompt = self.console.input("[bold blue]You:[/bold blue] ")
                    if prompt.lower() in ["exit", "quit"]:
                        break
                    if not prompt:
                        continue

                    await memory_manager.add_to_memory({"role": "user", "content": prompt})

                    context_messages = memory_manager.get_messages_for_context()

                    current_token_count = memory_manager.estimate_total_token_count(context_messages)
                    percent_used = (current_token_count / memory_manage.max_context_window) * 100
                    self.console.print(f"[dim]Context Usage: {current_token_count:,}/{memory_manager.max_context_window:,} tokens({percent_used:.1f}%)[/dim]")

                    result = await agent.run(user_prompt=promt, message_history=context_messages)

                    await memory_manager.add_to_memory({"role": "assistant", "content": result.output})

                    self.console.log(f"Usage: {result.usage()}")
                    self.console.log(Markdown(result.output))

                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
            self.console.print("\n[bold red]Exciting chat. Goodbye![/bold red]")
