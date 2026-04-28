from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class RequiredReagent(BaseModel):
    """Represents one reagent required to execute the protocol."""

    item_name: str = Field(..., min_length=2, description="Reagent or consumable name.")
    quantity_needed: str = Field(
        ...,
        min_length=1,
        description="Human-readable amount such as '500 mL' or '2 tubes'.",
    )
    purpose: str = Field(..., min_length=3, description="Why the reagent is needed.")

    @field_validator("item_name", "quantity_needed", "purpose")
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        """Normalize whitespace and reject empty-looking text values."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field value cannot be empty.")
        return cleaned


class ChronologicalStep(BaseModel):
    """Represents one ordered action in the protocol."""

    step_number: int = Field(..., ge=1, description="1-based sequence number.")
    action: str = Field(..., min_length=3, description="Concrete step instruction.")

    @field_validator("action")
    @classmethod
    def strip_action(cls, value: str) -> str:
        """Ensure step text is usable by downstream UI and parsing layers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Action cannot be empty.")
        return cleaned


class ExperimentProtocol(BaseModel):
    """Top-level schema for a structured experimental protocol."""

    experiment_title: str = Field(..., min_length=3, description="Short experiment name.")
    feasibility_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Estimated feasibility from 1 (low) to 10 (high).",
    )
    safety_hazards: List[str] = Field(
        default_factory=list,
        description="Known hazards, handling risks, or safety concerns.",
    )
    required_reagents: List[RequiredReagent] = Field(
        default_factory=list,
        description="Reagents or consumables needed for the experiment.",
    )
    chronological_steps: List[ChronologicalStep] = Field(
        default_factory=list,
        description="Ordered list of protocol actions.",
    )
    estimated_time_hours: float = Field(
        ...,
        gt=0,
        description="Estimated total experiment time in hours.",
    )

    @field_validator("experiment_title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        """Keep the title clean and reject blank input."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Experiment title cannot be empty.")
        return cleaned

    @field_validator("safety_hazards")
    @classmethod
    def clean_safety_hazards(cls, values: List[str]) -> List[str]:
        """Normalize hazard text and drop empty entries."""
        cleaned_values = [value.strip() for value in values if value and value.strip()]
        if not cleaned_values:
            raise ValueError("At least one safety hazard must be provided.")
        return cleaned_values

    @field_validator("required_reagents")
    @classmethod
    def require_reagents(cls, values: List[RequiredReagent]) -> List[RequiredReagent]:
        """Ensure the protocol contains at least one reagent."""
        if not values:
            raise ValueError("At least one required reagent must be provided.")
        return values

    @field_validator("chronological_steps")
    @classmethod
    def require_steps(cls, values: List[ChronologicalStep]) -> List[ChronologicalStep]:
        """Ensure the protocol contains at least one actionable step."""
        if not values:
            raise ValueError("At least one chronological step must be provided.")
        return values

    @model_validator(mode="after")
    def validate_step_order(self) -> "ExperimentProtocol":
        """Require strictly sequential step numbering starting at 1."""
        for expected_step_number, step in enumerate(self.chronological_steps, start=1):
            if step.step_number != expected_step_number:
                raise ValueError("chronological_steps must be sequential and start at 1.")
        return self
