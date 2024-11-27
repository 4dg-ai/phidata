from abc import ABC, abstractmethod
from typing import Optional, List

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:
    raise ImportError("`pymongo` not installed. Please install it with `pip install pymongo`")

from phi.memory.db import MemoryDb
from phi.memory.row import MemoryRow
from phi.utils.log import logger


class MongoDbMemoryDb(MemoryDb):
    def __init__(
        self,
        db_name: str = "memory_db",
        collection_name: str = "memory",
        db_url: Optional[str] = None,
        client: Optional[MongoClient] = None,
    ):
        """
        This class provides a memory store backed by a MongoDB collection.

        The following order is used to determine the database connection:
            1. Use the client if provided
            2. Use the db_url

        Args:
            db_name: The name of the database to store Agent sessions.
            collection_name: The name of the collection to store Agent sessions.
            db_url: The database URL to connect to.
            client: The MongoDB client to use.
        """
        _client: Optional[MongoClient] = client
        if _client is None and db_url is not None:
            _client = MongoClient(db_url)
        elif _client is None:
            raise ValueError("Must provide either db_url or client")

        # Database attributes
        self.client: MongoClient = _client
        self.db_name: str = db_name
        self.collection_name: str = collection_name
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def create(self) -> None:
        """
        Ensure the collection exists.
        """
        try:
            if self.collection_name not in self.db.list_collection_names():
                logger.debug(f"Creating collection: {self.collection_name}")
                self.db.create_collection(self.collection_name)
        except PyMongoError as e:
            logger.error(f"Error creating collection '{self.collection_name}': {e}")
            raise

    def memory_exists(self, memory: MemoryRow) -> bool:
        try:
            result = self.collection.find_one({"_id": memory.id})
            return result is not None
        except PyMongoError as e:
            logger.error(f"Error checking if memory exists: {e}")
            return False

    def read_memories(
        self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories: List[MemoryRow] = []
        try:
            query = {}
            if user_id is not None:
                query["user_id"] = user_id

            cursor = self.collection.find(query)

            if sort == "asc":
                cursor = cursor.sort("created_at", 1)
            else:
                cursor = cursor.sort("created_at", -1)

            if limit is not None:
                cursor = cursor.limit(limit)

            for document in cursor:
                memories.append(MemoryRow(id=document["_id"], user_id=document["user_id"], memory=document["memory"]))
        except PyMongoError as e:
            logger.error(f"Exception reading from collection: {e}")
        return memories

    def upsert_memory(self, memory: MemoryRow, create_and_retry: bool = True) -> None:
        try:
            self.collection.update_one(
                {"_id": memory.id},
                {"$set": {"user_id": memory.user_id, "memory": memory.memory}},
                upsert=True,
            )
        except PyMongoError as e:
            logger.error(f"Exception upserting into collection: {e}")
            if create_and_retry:
                logger.info(f"Collection does not exist: {self.collection_name}")
                logger.info("Creating collection for future transactions")
                self.create()
                return self.upsert_memory(memory, create_and_retry=False)
            else:
                raise

    def delete_memory(self, id: str) -> None:
        try:
            self.collection.delete_one({"_id": id})
        except PyMongoError as e:
            logger.error(f"Exception deleting memory: {e}")

    def drop_table(self) -> None:
        try:
            if self.collection_name in self.db.list_collection_names():
                logger.debug(f"Deleting collection: {self.collection_name}")
                self.collection.drop()
        except PyMongoError as e:
            logger.error(f"Error dropping collection '{self.collection_name}': {e}")

    def table_exists(self) -> bool:
        logger.debug(f"Checking if collection exists: {self.collection_name}")
        try:
            return self.collection_name in self.db.list_collection_names()
        except PyMongoError as e:
            logger.error(e)
            return False

    def clear(self) -> bool:
        try:
            self.collection.delete_many({})
            return True
        except PyMongoError as e:
            logger.error(f"Exception clearing collection: {e}")
            return False