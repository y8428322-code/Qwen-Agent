# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for tool-call preservation in the OpenAI-compatible path.

Both cases below are reachable from ordinary agent usage and are rejected by
strict OpenAI-compatible endpoints:

* parallel tool calls that share one id -> ``400 invalid params, duplicate
  tool_call id: 1 (2013)``
* a non-streaming reply that carries both ``reasoning_content`` and
  ``tool_calls`` -> the tool calls are silently dropped, so the agent never
  performs the requested action.
"""

from types import SimpleNamespace
from unittest import mock

from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.oai import TextChatAtOAI
from qwen_agent.llm.schema import Message


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, type='function',
                           function=SimpleNamespace(name=name, arguments=arguments))


def test_parallel_tool_calls_get_unique_ids():
    """Two tool calls in one assistant message must not share an id."""
    messages = [
        Message(role='user', content='hi'),
        Message(role='assistant', content='', function_call={'name': 'f', 'arguments': '{}'}),
        Message(role='assistant', content='', function_call={'name': 'g', 'arguments': '{}'}),
    ]

    converted = BaseChatModel._conv_qwen_agent_messages_to_oai(messages)

    tool_calls = converted[1]['tool_calls']
    assert len(tool_calls) == 2
    ids = [tc['id'] for tc in tool_calls]
    assert len(set(ids)) == 2, f'tool_call ids must be unique, got {ids}'


def test_explicit_function_ids_are_preserved():
    """An explicit ``extra.function_id`` must still be used as the tool_call id."""
    messages = [
        Message(role='user', content='hi'),
        Message(role='assistant',
                content='',
                function_call={'name': 'f', 'arguments': '{}'},
                extra={'function_id': 'call_abc'}),
    ]

    converted = BaseChatModel._conv_qwen_agent_messages_to_oai(messages)

    assert converted[1]['tool_calls'][0]['id'] == 'call_abc'


def test_no_stream_keeps_tool_calls_when_reasoning_content_present():
    """A reply with both ``reasoning_content`` and ``tool_calls`` must keep both."""
    llm = TextChatAtOAI({'model': 'some-model', 'model_type': 'oai', 'api_key': 'k'})

    fake_message = SimpleNamespace(
        content='',
        reasoning_content='let me think...',
        tool_calls=[_tool_call('call_1', 'f', '{"a": 1}')],
    )
    fake_response = SimpleNamespace(choices=[SimpleNamespace(message=fake_message)])

    # `_chat_complete_create` 是实例属性（在 __init__ 里赋值），因此在实例上替换
    llm._chat_complete_create = mock.Mock(return_value=fake_response)

    out = llm._chat_no_stream([Message(role='user', content='hi')], {})

    assert any(getattr(m, 'function_call', None) is not None for m in out), (
        'tool_calls were dropped by the reasoning_content branch')
    assert any(getattr(m, 'reasoning_content', None) for m in out)
