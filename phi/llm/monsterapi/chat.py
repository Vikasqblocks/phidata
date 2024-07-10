import os
import httpx

from typing import Dict, Any

from phi.utils.log import logger
from phi.llm.openai.chat import OpenAIChat

try:
    from openai import OpenAI as OpenAIClient, AsyncOpenAI as AsyncOpenAIClient
except ImportError:
    logger.error("`openai` not installed")
    raise

class MonsterAPIChat(OpenAIChat):
    """
    Integrating MonsterAPI LLMs into phidata

    Details about MonsterAPI can be found here: https://llm.monsterapi.ai/docs

    Just import object and set MONSTER_API_KEY from https://monsterapi.ai/user/dashboard

    How to use it ?
    ---------------
    ```python3
    from phi.assistant import Assistant
    from phi.llm.monsterapi import MonsterAPIChat
    from phi.tools.yfinance import YFinanceTools

    assistant = Assistant(
        llm=MonsterAPIChat(model="meta-llama/Meta-Llama-3-8B-Instruct"),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
        show_tool_calls=True,
        markdown=True,
    )
    assistant.print_response("What is the stock price of NVDA")
    assistant.print_response("Write a comparison between NVDA and AMD, use all tools available.")
    ```
    """
    name: str = "MonsterAPIChat"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    base_url: str = "https://llm.monsterapi.ai/v1/"

    def get_client(self) -> OpenAIClient:
        if self.client:
            return self.client

        if self.openai_client:
            return self.openai_client

        _client_params: Dict[str, Any] = {}
        api_key = self.api_key or os.getenv("MONSTER_API_KEY")
        if api_key:
            _client_params["api_key"] = api_key
        
        _client_params["organization"] = None
        _client_params["base_url"] = self.base_url
        if self.timeout:
            _client_params["timeout"] = self.timeout
        if self.max_retries:
            _client_params["max_retries"] = self.max_retries
        if self.default_headers:
            _client_params["default_headers"] = self.default_headers
        if self.default_query:
            _client_params["default_query"] = self.default_query
        if self.http_client:
            _client_params["http_client"] = self.http_client
        if self.client_params:
            _client_params.update(self.client_params)
        return OpenAIClient(**_client_params)

    def get_async_client(self) -> AsyncOpenAIClient:
        if self.async_client:
            return self.async_client

        _client_params: Dict[str, Any] = {}
        api_key = self.api_key or os.getenv("MONSTER_API_KEY")
        if api_key:
            _client_params["api_key"] = api_key
        
        _client_params["organization"] = None
        _client_params["base_url"] = self.base_url
        if self.timeout:
            _client_params["timeout"] = self.timeout
        if self.max_retries:
            _client_params["max_retries"] = self.max_retries
        if self.default_headers:
            _client_params["default_headers"] = self.default_headers
        if self.default_query:
            _client_params["default_query"] = self.default_query
        if self.http_client:
            _client_params["http_client"] = self.http_client
        else:
            _client_params["http_client"] = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            )
        if self.client_params:
            _client_params.update(self.client_params)
        return AsyncOpenAIClient(**_client_params)