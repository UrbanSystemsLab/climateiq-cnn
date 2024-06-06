import json
from google.cloud import firestore
from settings import Settings

"""
    This class is used to query Firestore metastore.

    Input parameters:
        firestore_client: A Firestore client object.
        settings: A Settings object.

    Output parameters:
        None

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

    def _print_document_content(collection_info, document_id):
        """Prints the content of a document if it exists in the collection_info.

        Args:
            collection_info: A dictionary representing the collection information.
            document_id: The ID of the document to print.
        """
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

    def _find_simulation_document(self, collections_info, study_area):
        """ Finds the simulation document for a given study area. """
        print("Finding simulation document...")
        for collection_name, collection_data in collections_info.items():
            if collection_name == "simulations":
                for document_id, subcollections in collection_data.items():
                    if study_area in document_id:
                        return {
                            "document_id": document_id,
                            "subcollections": subcollections,
                        }
        return None

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

    def _extract_rainfall_info(self):
        """
        Extract rainfall information from Firestore.
        """
        print("Extracting rainfall info...")
        collection_name = "city_cat_rainfall_configs"
        try:
            print(f"Querying Firestore collection: {collection_name}")
            docs = self.firestore_client.collection(collection_name).stream()
            print("Successfully queried Firestore collection")
        except Exception as e:
            print(f"Error querying Firestore: {e}")
            return []

        rainfall_info = []

        for doc in docs:
            try:
                data = doc.to_dict()
                if "rainfall_duration" in data and "as_vector_gcs_uri" in data:
                    info = {
                        "rainfall_duration": data["rainfall_duration"],
                        "as_vector_gcs_uri": data["as_vector_gcs_uri"],
                        "parent_config_name": data["parent_config_name"],
                    }
                    rainfall_info.append(info)
                    print(f"Extracted info: {info}")
            except Exception as e:
                print(f"Error processing document {doc.id}: {e}")

        return rainfall_info

    def _get_rainfall_durations(self, collection_name="city_cat_rainfall_configs"):
        """
        Get rainfall durations from Firestore.
        """
        print("Getting rainfall durations...")
        try:
            print(f"Querying Firestore collection: {collection_name}")
            docs = self.firestore_client.collection(collection_name).stream()
            print("Successfully queried Firestore collection.")
        except Exception as e:
            print(f"Error querying Firestore: {e}")
            return []

        rainfall_durations = [
            doc.to_dict().get("rainfall_duration")
            for doc in docs
            if "rainfall_duration" in doc.to_dict()
        ]
        print(f"Found rainfall durations: {rainfall_durations}")
        return rainfall_durations

    def _get_label_chunks_urls(self, document_id):
        """
        Get the URLs of the label chunks for a given document ID.
        """
        print(f"Getting label chunks URLs for document: {document_id}")
        # Use the document_id directly as provided
        collection_path = f"simulations/{document_id}/label_chunks"
        try:
            print(f"Querying Firestore collection: {collection_path}")
            docs = self.firestore_client.collection(collection_path).stream()
            print("Successfully queried Firestore collection.")
        except Exception as e:
            print(f"Error querying Firestore: {e}")
            return []

        gcs_urls = []
        for doc in docs:
            if self.settings.DEBUG == 2:
                print(f"Found document: {doc.id}, data: {doc.to_dict()}")
            gcs_uri = doc.to_dict().get("gcs_uri")
            if gcs_uri:
                gcs_urls.append(gcs_uri)
                if self.settings.DEBUG == 2:
                    print(f"Found GCS URI: {gcs_uri}")

        return gcs_urls
