from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from .inference import MODEL_ID


class ChatMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def require_nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must contain non-whitespace text.")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    stream: bool = False

    @field_validator("model")
    @classmethod
    def require_supported_model(cls, value: str) -> str:
        if value != MODEL_ID:
            raise ValueError(f"The supported model is {MODEL_ID}.")
        return value

    @model_validator(mode="after")
    def require_supported_settings(self) -> Self:
        if self.temperature == 0 and self.top_p != 1:
            raise PydanticCustomError(
                "invalid_parameter_combination",
                "Greedy decoding requires top_p=1.0.",
            )
        if self.stream:
            raise PydanticCustomError(
                "unsupported_parameter",
                "Streaming is not supported yet.",
            )
        return self
