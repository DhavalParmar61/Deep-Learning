from typing import Any, List, Optional, Dict

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[str]]


class MultipleHouseDataInputs(BaseModel):
    inputs: Dict

    class Config:
        schema_extra = {
            "example": {
                    "inputs":{
                    "sentences1": [
                        "This church choir sings to the masses as they sing joyous songs from the book at a church.",
                        "This church choir sings to the masses as they sing joyous songs from the book at a church.",
                    ],
                    "sentences2": [
                        "The church has cracks in the ceiling.",
                        "The church is filled with song.",
                    ]
                    }
            }
        }