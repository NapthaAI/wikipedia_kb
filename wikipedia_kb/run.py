import logging
import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from naptha_sdk.schemas import KBDeployment
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.storage.schemas import CreateTableRequest, CreateRowRequest, ReadStorageRequest, DeleteStorageRequest, ListStorageRequest, DatabaseReadOptions

from wikipedia_kb.schemas import InputSchema

logger = logging.getLogger(__name__)


class WikipediaKB:
    def __init__(self, deployment: Dict[str, Any]):
        self.deployment = deployment
        self.config = self.deployment.config
        self.storage_provider = StorageProvider(self.deployment.node)
        self.storage_type = self.config["storage_type"]
        self.table_name = self.config["path"]
        self.schema = self.config["schema"]
    
    # TODO: Remove this. In future, the create function should be called by create_module in the same way that run is called by run_module
    async def init(self, *args, **kwargs):
        await create(self.deployment)
        return {"status": "success", "message": f"Successfully populated {self.table_name} table"}
    
    async def add_data(self, *args, **kwargs):

        # Add rows to the table
        data = json.loads(self.input_schema.data)

        logger.info(f"Adding {len(data)} rows to table {self.table_name}")
        for row in tqdm(data, total=len(data)):
            # if row has no id, generate a random one
            if 'id' not in row:
                row['id'] = random.randint(1, 1000000)

            read_result = await self.storage_provider.read(
                storage_type=self.storage_type,
                path=self.table_name,
                condition={'title': row['title']}
            )

            # make sure title are not in the table
            if read_result["rows"]:
                continue

            create_row_result = await self.storage_provider.create(CreateRowRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                data=row
            ))
            logger.info(f"Create row result: {create_row_result}")

        logger.info(f"Successfully added {len(data)} rows to table {self.table_name}")
        return {"status": "success", "message": f"Successfully added {len(data)} rows to table {self.table_name}"}

    async def run_query(self, input_data: Dict[str, Any], *args, **kwargs):
        logger.info(f"Querying table {self.table_name} with query: {input_data['query']}")

        read_storage_request = ReadStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options=DatabaseReadOptions(conditions=[{'title': input_data['query']}])
        )

        read_result = await self.storage_provider.read(read_storage_request)
        logger.info(f"Query results: {read_result}")
        return {"status": "success", "message": f"Query results: {read_result}"}

    async def list_rows(self, input_data: Dict[str, Any], *args, **kwargs):
        list_storage_request = ListStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options=DatabaseReadOptions(limit=input_data['limit'])
        )
        list_storage_result = await self.storage_provider.list(list_storage_request)
        logger.info(f"List rows result: {list_storage_result}")
        return {"status": "success", "message": f"List rows result: {list_storage_result}"}

    async def delete_table(self, input_data: Dict[str, Any], *args, **kwargs):
        delete_table_request = DeleteStorageRequest(
            storage_type=self.storage_type,
            path=input_data['table_name'],
        )
        delete_table_result = await self.storage_provider.delete(delete_table_request)
        logger.info(f"Delete table result: {delete_table_result}")
        return {"status": "success", "message": f"Delete table result: {delete_table_result}"}

    async def delete_row(self, input_data: Dict[str, Any], *args, **kwargs):
        delete_row_request = DeleteStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options={"condition": input_data['condition']}
        )

        delete_row_result = await self.storage_provider.delete(delete_row_request)
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
    storage_type = deployment.config["storage_type"]
    table_name = deployment.config["path"]
    schema = deployment.config["schema"]

    logger.info(f"Creating {storage_type} at {table_name} with schema {schema}")

    create_table_request = CreateTableRequest(
        storage_type=storage_type,
        path=table_name,
        schema=schema
    )

    # Create a table
    create_table_result = await storage_provider.create(create_table_request)

    logger.info(f"Result: {create_table_result}")

    # Load the df
    df = pd.read_parquet(file_path)

    # Add rows to the table
    logger.info("Adding rows to table")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_data = {
            "id": int(row['id']),
            "url": str(row['url']),
            "title": str(row['title']),
            "text": str(row['text'])
        }

        # Add a row
        create_row_result = await storage_provider.create(CreateRowRequest(
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
    from naptha_sdk.schemas import KBRunInput

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("kb", "wikipedia_kb/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    inputs_dict = {
        "init": InputSchema(
            function_name="init",
            function_input_data=None,
        ),
        "add_data": InputSchema(
            function_name="add_data",
            function_input_data=None,
        ),
        "run_query": InputSchema(
            function_name="run_query",
            function_input_data={"query": "Elon Musk"},
        ),
        "list_rows": InputSchema(
            function_name="list_rows",
            function_input_data={"limit": 10},
        ),
        "delete_table": InputSchema(
            function_name="delete_table",
            function_input_data={"table_name": "wikipedia_kb"},
        ),
        "delete_row": InputSchema(
            function_name="delete_row",
            function_input_data={"condition": {"title": "Elon Musk"}},
        ),
    }

    module_run = KBRunInput(
        inputs=inputs_dict["init"],
        deployment=deployment,
        consumer_id=naptha.user.id,
    )

    response = asyncio.run(run(module_run))

    print("Response", response)