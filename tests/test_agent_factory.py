from pier.agents.factory import AgentFactory
from pier.models.agent.name import AgentName
from pier.models.trial.config import AgentConfig


def test_mini_swe_agent_supports_request_throttling():
    config = AgentConfig(name=AgentName.MINI_SWE_AGENT.value)

    assert AgentFactory.supports_request_throttling(config)


def test_agent_without_throttling_capability_returns_false():
    config = AgentConfig(name=AgentName.NOP.value)

    assert not AgentFactory.supports_request_throttling(config)


def test_custom_agent_does_not_support_request_throttling():
    config = AgentConfig(import_path="custom_agent:CustomAgent")

    assert not AgentFactory.supports_request_throttling(config)
