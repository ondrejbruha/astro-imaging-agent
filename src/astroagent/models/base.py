from pydantic import BaseModel, ConfigDict


class SchemaModel(BaseModel):
    """Reject misspelled fields and nonfinite numbers in serialized models."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
