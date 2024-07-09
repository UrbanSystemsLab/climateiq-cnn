import json
from google.cloud import firestore
from usl_models.flood_ml.settings import Settings


"""
    This class is used to query Firestore metastore.

"""


class FirestoreDataHandler:
    def __init__(
        self,
        firestore_client: firestore.Client = None,
        settings: Settings = None,
    ):
        print("Initializing FirestoreDataHandler...")
        self.firestore_client = firestore_client or firestore.Client()

        # Load settings
        self.settings = settings or Settings()

    def _fetch_document(self, document_ref):
        """
        Fetches a document from Firestore using the provided document reference.

        Args:
            document_ref: The reference of the document to fetch.

        Returns:
            The fetched document as a dictionary.
        """
        try:
            print(f"Fetching document: {document_ref.id}")
            # Fetch the document
            document = document_ref.get()

            # Return the document data as a dictionary
            return document.to_dict()
        except Exception as e:
            print(f"An error occurred while fetching the document: {e}")
            return None

    def _find_document_by_id(self, collection_name, document_id):
        """
        Finds a document by ID in the specified collection.

        Args:
            collection_name: The name of the collection to search in.
            document_id: The ID of the document to find.

        Returns:
            The found document as a dictionary, or None if no document was found.
        """
        try:
            if self.settings.DEBUG > 2:
                print(f"Searching for document with ID '{document_id}' in collection '{collection_name}'")

            # Get the document with the given ID
            document = self.firestore_client.collection(collection_name).document(document_id).get()

            # If the document exists, return it as a dictionary
            if document.exists:
                if self.settings.DEBUG > 2:
                    print(f"Found document: {document.to_dict()}")
                return document.to_dict()
            else:
                print(f"No document found with ID '{document_id}' in collection '{collection_name}'")
                return None

        except Exception as e:
            print(f"An error occurred while finding the document: {e}")
            return None

    def _print_document_content(self, collection_info, document_id):
        """Prints the content of a document if it exists in the collection_info.

        Args:
            collection_info: A dictionary representing the collection information.
            document_id: The ID of the document to print.
        """
        print("Printing document content...")
        print(f"Collection: {collection_info}")
        print(f"Document ID: {document_id}")

        for collection_name, collection_data in collection_info.items():
            if document_id in collection_data:
                print(f"Document ID: {document_id}")
                print(json.dumps(collection_data[document_id], indent=4))
                return

        print(f"Document with ID '{document_id}' not found in the collection.")

    def _is_document_in_collection(self, collection_info, object_name):
        """Checks if an object is a document."""
        for collection_name, collection_data in collection_info.items():
            if object_name in collection_data:
                return True
        return False

    def _find_simulation_documents(self, simulation_collection_name):
        """Finds all documents under a given simulation collection."""
        print(f"Finding documents under collection: {simulation_collection_name}")
        simulation_documents = []

        # Get the collection
        collection_ref = self.firestore_client.collection(simulation_collection_name)

        # Get all documents from the collection
        docs = collection_ref.stream()

        if not docs:
            print(f"No documents found under collection: {simulation_collection_name}")
            return simulation_documents
        else :
            print(f"Found documents under collection: {simulation_collection_name}")
    
            for doc in docs:
                document_data = doc.to_dict()
                document_id = doc.id

                # Get all subcollections of the document
                subcollections = (
                    self.firestore_client.collection(simulation_collection_name)
                    .document(document_id)
                    .collections()
                )

                subcollections_data = {}
                for subcollection in subcollections:
                    subcollection_docs = subcollection.stream()
                    subcollections_data[subcollection.id] = [
                        doc.to_dict() for doc in subcollection_docs
                    ]

                simulation_documents.append(
                    {
                        "document_id": document_id,
                        "document_data": document_data,
                        "subcollections": subcollections_data,
                    }
                )

        return simulation_documents

    def get_documents(self, collection_name):
        """
        Get all documents in a collection.
        """
        print(f"Getting documents in collection: {collection_name}")
        documents = []

        # Get the collection
        collection_ref = self.firestore_client.collection(collection_name)

        # Get all documents from the collection
        docs = collection_ref.stream()

        for doc in docs:
            # Append the document data as a dictionary to the list
            documents.append({
                "document_id": doc.id,  # Include the document ID
                "document_data": doc.to_dict()  # Include the document data
            })

        return documents

    def _list_collections_and_print_documents(self):
        """
        List all collections and print the documents in each collection.
        """
        print("Listing collections and documents...")
        collections_info = {}

        # List all top-level collections
        collections = self.firestore_client.collections()
        for collection in collections:
            collection_name = collection.id
            collections_info[collection_name] = {}

            # List all documents in the collection
            docs = collection.stream()
            for doc in docs:
                doc_id = doc.id
                collections_info[collection_name][doc_id] = []

                # Print any document in the collection
                # print(f"Collection: {collection_name}")
                # print(f"  Document ID: {doc_id}")
                # print(f"  Document Data: {doc.to_dict()}")

                # List all subcollections for each document
                subcollections = (
                    self.firestore_client.collection(collection_name)
                    .document(doc_id)
                    .collections()
                )
                for subcollection in subcollections:
                    subcollection_name = subcollection.id
                    collections_info[collection_name][doc_id].append(subcollection_name)

                break

        return collections_info