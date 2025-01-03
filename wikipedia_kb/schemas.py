from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from naptha_sdk.schemas import KBConfig

class InputSchema(BaseModel):
    function_name: Literal["init", "run_query", "add_data", "delete_table", "delete_row", "list_rows"]
    function_input_data: Optional[Dict[str, Any]] = None
