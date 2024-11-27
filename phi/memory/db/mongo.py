from typing import Optional, List, Dict

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    from bson import ObjectId
except ImportError:
    raise ImportError("`pymongo` not installed. Please install it with `pip install pymongo")

from phi.utils.log import logger


class MongoDb:
    def __init__(
        self,
        db_name: str = "memory_db",
        collection_name: str = "memory",
        db_url: Optional[str] = None,
        client: Optional[MongoClient] = None,
    ):
        """
        This class provides a reader for a MongoDB collection.

        The following order is used to determine the database connection:
            1. Use the client if provided
            2. Use the db_url

        Args:
            db_name: The name of the database to connect to.
            collection_name: The name of the collection to read from.
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

    def read_documents(
        self, query: Optional[Dict] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[Dict]:
        """
        Read documents from the MongoDB collection based on the provided query.

        Args:
            query: The query to filter documents.
            limit: The maximum number of documents to retrieve.
            sort: The sorting order for the documents ("asc" or "desc").

        Returns:
            A list of documents matching the query.
        """
        documents: List[Dict] = []
        try:
            if query is None:
                query = {}

            cursor = self.collection.find(query)

            if sort == "asc":
                cursor = cursor.sort("created_at", 1)
            elif sort == "desc":
                cursor = cursor.sort("created_at", -1)

            if limit is not None:
                cursor = cursor.limit(limit)

            for document in cursor:
                documents.append(document)
        except PyMongoError as e:
            logger.error(f"Exception reading from collection: {e}")
        return documents

    def get_schema(self) -> Dict[str, str]:
        """
        Get the schema of the MongoDB collection by analyzing the first document.

        Returns:
            A dictionary representing the schema, where keys are field names and values are field types.
        """
        schema: Dict[str, str] = {}
        try:
            document = self.collection.find_one()
            if document:
                for key, value in document.items():
                    schema[key] = type(value).__name__
        except PyMongoError as e:
            logger.error(f"Exception getting schema from collection: {e}")
        return schema

    def drop_collection(self) -> None:
        """
        Drop the collection if it exists.
        """
        try:
            if self.collection_name in self.db.list_collection_names():
                logger.debug(f"Deleting collection: {self.collection_name}")
                self.collection.drop()
        except PyMongoError as e:
            logger.error(f"Error dropping collection '{self.collection_name}': {e}")

    def collection_exists(self) -> bool:
        """
        Check if the collection exists in the database.

        Returns:
            True if the collection exists, False otherwise.
        """
        logger.debug(f"Checking if collection exists: {self.collection_name}")
        try:
            return self.collection_name in self.db.list_collection_names()
        except PyMongoError as e:
            logger.error(e)
            return False

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if the collection was cleared successfully, False otherwise.
        """
        try:
            self.collection.delete_many({})
            return True
        except PyMongoError as e:
            logger.error(f"Exception clearing collection: {e}")
            return False