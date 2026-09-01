"""Runtime settings loaded from environment variables."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Langfuse is the default tracer (MIT, self-hostable). See agent_team.tracing.
    langfuse_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "langfuse_enabled",
            "LANGFUSE_ENABLED",
            "langfuse_tracing",
            "LANGFUSE_TRACING",
        ),
    )
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = Field(
        default="",
        validation_alias=AliasChoices(
            "langfuse_host",
            "LANGFUSE_HOST",
            "langfuse_base_url",
            "LANGFUSE_BASE_URL",
        ),
    )

    # MLflow is the experiment-platform alternative when Langfuse keys are absent.
    mlflow_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "mlflow_enabled",
            "MLFLOW_ENABLED",
            "mlflow_tracing",
            "MLFLOW_TRACING",
        ),
    )
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "agent-team"
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""
    mlflow_tracking_token: str = ""

    # LangSmith is the LangChain-native alternative when Langfuse and MLflow are absent.
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "langsmith_tracing",
            "LANGSMITH_TRACING",
            "langchain_tracing_v2",
            "LANGCHAIN_TRACING_V2",
        ),
    )
    langsmith_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "langsmith_api_key",
            "LANGSMITH_API_KEY",
            "langchain_api_key",
            "LANGCHAIN_API_KEY",
        ),
    )
    langsmith_project: str = Field(
        default="multi-agent-slack-pipeline",
        validation_alias=AliasChoices(
            "langsmith_project",
            "LANGSMITH_PROJECT",
            "langchain_project",
            "LANGCHAIN_PROJECT",
        ),
    )

    langgraph_server_url: str = "http://localhost:8123"
    langgraph_assistant_id: str = "agent_team"

    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_channel: str = "#agent-approvals"
    slack_enabled: bool = False
    slack_api_base_url: str = "https://slack.com/api/"
    slack_interactivity_url: str = "http://localhost:8080/slack/interactive"
    slack_emulator_host: str = "0.0.0.0"
    slack_emulator_port: int = 8081
    slack_emulator_auto_approve: bool = True
    slack_emulator_approve_delay_seconds: float = 0.4

    nats_enabled: bool = False
    nats_url: str = "nats://localhost:4222"
    nats_stream: str = "AGENT_REVIEW"
    nats_subject: str = "agent.review.required"
    nats_queue: str = Field(
        default="slack-publisher",
        validation_alias=AliasChoices("nats_queue", "NATS_QUEUE", "nats_durable", "NATS_DURABLE"),
    )
    nats_feedback_subject: str = "agent.review.feedback"
    nats_feedback_queue: str = "workflow-resumer"

    auto_approve: bool = False
    max_supervisor_steps: int = 12

    slack_gateway_host: str = "0.0.0.0"
    slack_gateway_port: int = 8080

    # OpenTelemetry for the Slack gateway and NATS workers. Endpoints and
    # tokens stay in env so each process can target a local collector, a
    # remote collector, or a vendor OTLP intake.
    otel_sdk_disabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("otel_sdk_disabled", "OTEL_SDK_DISABLED"),
    )
    otel_service_name: str = Field(
        default="",
        validation_alias=AliasChoices("otel_service_name", "OTEL_SERVICE_NAME"),
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices(
            "otel_exporter_otlp_endpoint",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
        ),
    )
    otel_exporter_otlp_protocol: str = Field(
        default="http/protobuf",
        validation_alias=AliasChoices(
            "otel_exporter_otlp_protocol",
            "OTEL_EXPORTER_OTLP_PROTOCOL",
        ),
    )
    otel_exporter_otlp_headers: str = Field(
        default="",
        validation_alias=AliasChoices(
            "otel_exporter_otlp_headers",
            "OTEL_EXPORTER_OTLP_HEADERS",
        ),
    )
    otel_resource_attributes: str = Field(
        default="",
        validation_alias=AliasChoices(
            "otel_resource_attributes",
            "OTEL_RESOURCE_ATTRIBUTES",
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
