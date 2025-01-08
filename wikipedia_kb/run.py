import logging
import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List
from naptha_sdk.schemas import KBDeployment, KBRunInput
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.storage.schemas import CreateStorageRequest, ReadStorageRequest, DeleteStorageRequest, ListStorageRequest, DatabaseReadOptions
from naptha_sdk.user import sign_consumer_id

from wikipedia_kb.schemas import InputSchema

logger = logging.getLogger(__name__)


class WikipediaKB:
    def __init__(self, deployment: Dict[str, Any]):
        self.deployment = deployment
        self.config = self.deployment.config
        self.storage_provider = StorageProvider(self.deployment.node)
        self.storage_type = self.config.storage_type
        self.table_name = self.config.path
        self.schema = self.config.schema
    
    # TODO: Remove this. In future, the create function should be called by create_module in the same way that run is called by run_module
    async def init(self, *args, **kwargs):
        await create(self.deployment)
        return {"status": "success", "message": f"Successfully populated {self.table_name} table"}
    
    async def add_data(self, input_data: Dict[str, Any], *args, **kwargs):
        logger.info(f"Adding {(input_data)} to table {self.table_name}")

        # if row has no id, generate a random one
        if 'id' not in input_data:
            input_data['id'] = random.randint(1, 1000000)

        # read_result = await self.storage_provider.execute(ReadStorageRequest(
        #     storage_type=self.storage_type,
        #     path=self.table_name,
        #     options={"condition": {"title": input_data["title"]}}
        # ))

        # make sure title are not in the table
        # if len(read_result) > 0:
        #     return {"status": "error", "message": f"Title {input_data['title']} already exists in table {self.table_name}"}

        create_row_result = await self.storage_provider.execute(CreateStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            data={"data": input_data}
        ))
        logger.info(f"Create row result: {create_row_result}")

        logger.info(f"Successfully added {input_data} to table {self.table_name}")
        return {"status": "success", "message": f"Successfully added {input_data} to table {self.table_name}"}

    async def run_query(self, input_data: Dict[str, Any], *args, **kwargs):
        logger.info(f"Querying table {self.table_name} with query: {input_data['query']}")

        read_storage_request = ReadStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options={"condition": {"title": input_data["query"]}}
        )

        read_result = await self.storage_provider.execute(read_storage_request)
        logger.info(f"Query results: {read_result}")
        return {"status": "success", "message": f"Query results: {read_result}"}

    async def list_rows(self, input_data: Dict[str, Any], *args, **kwargs):
        list_storage_request = ListStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options={"limit": input_data['limit'] if input_data and 'limit' in input_data else None}
        )
        list_storage_result = await self.storage_provider.execute(list_storage_request)
        logger.info(f"List rows result: {list_storage_result}")
        return {"status": "success", "message": f"List rows result: {list_storage_result}"}

    async def delete_table(self, input_data: Dict[str, Any], *args, **kwargs):
        delete_table_request = DeleteStorageRequest(
            storage_type=self.storage_type,
            path=input_data['table_name'],
        )
        delete_table_result = await self.storage_provider.execute(delete_table_request)
        logger.info(f"Delete table result: {delete_table_result}")
        return {"status": "success", "message": f"Delete table result: {delete_table_result}"}

    async def delete_row(self, input_data: Dict[str, Any], *args, **kwargs):
        delete_row_request = DeleteStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options={"condition": input_data['condition']}
        )

        delete_row_result = await self.storage_provider.execute(delete_row_request)
        logger.info(f"Delete row result: {delete_row_result}")
        return {"status": "success", "message": f"Delete row result: {delete_row_result}"}

# TODO: Make it so that the create function is called when the kb/create endpoint is called
async def create(deployment: KBDeployment):
    """
    Create the Wikipedia Knowledge Base table
    Args:
        deployment: Deployment configuration containing deployment details
    """
    file_path = Path(__file__).parent / "data" / "wikipedia_kb_sample.parquet"

    storage_provider = StorageProvider(deployment.node)
    storage_type = deployment.config.storage_type
    table_name = deployment.config.path
    schema = {"schema": deployment.config.schema}

    logger.info(f"Creating {storage_type} at {table_name} with schema {schema}")


    create_table_request = CreateStorageRequest(
        storage_type=storage_type,
        path=table_name,
        data=schema
    )

    # Create a table
    create_table_result = await storage_provider.execute(create_table_request)

    logger.info(f"Result: {create_table_result}")

    # Load the df
    df = pd.read_parquet(file_path)

    # Add rows to the table
    logger.info("Adding rows to table")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_data = {
            "data": {
                "id": int(row['id']),
                "url": str(row['url']),
                "title": str(row['title']),
                "text": str(row['text'])
            }
        }

        # Add a row
        create_row_result = await storage_provider.execute(CreateStorageRequest(
            storage_type=storage_type,
            path=table_name,
            data=row_data
        ))

        logger.info(f"Add row result: {create_row_result}")
    
    logger.info(f"Successfully populated {table_name} table with {len(df)} rows")


async def run(module_run: Dict[str, Any], *args, **kwargs):
    """
    Run the Wikipedia Knowledge Base deployment
    Args:
        module_run: Module run configuration containing deployment details
    """
    module_run = KBRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    wikipedia_kb = WikipediaKB(module_run.deployment)

    method = getattr(wikipedia_kb, module_run.inputs.function_name, None)

    if not method:
        raise ValueError(f"Invalid function name: {module_run.inputs.function_name}")

    return await method(module_run.inputs.function_input_data)


if __name__ == "__main__":
    import asyncio
    import os
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("kb", "wikipedia_kb/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    inputs_dict = {
        "init": {
            "function_name": "init",
            "function_input_data": None,
        },
        "add_data": {
            "function_name": "add_data",
            "function_input_data": {
                "url": "https://en.wikipedia.org/wiki/Socrates",
                "title": "Socrates", 
                "text": "Socrates was a Greek philosopher from Athens who is credited as the founder of Western philosophy and as among the first moral philosophers of the ethical tradition of thought."
            },
        },
        "run_query": {
            "function_name": "run_query",
            "function_input_data": {"query": "Elon Musk"},
        },
        "list_rows": {
            "function_name": "list_rows",
            "function_input_data": {"limit": 10},
        },
        "delete_table": {
            "function_name": "delete_table",
            "function_input_data": {"table_name": "wikipedia_kb"},
        },
        "delete_row": {
            "function_name": "delete_row",
            "function_input_data": {"condition": {"title": "Elon Musk"}},
        },
    }

    module_run = {
        "inputs": inputs_dict["delete_table"],
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(module_run))

    print("Response", response)