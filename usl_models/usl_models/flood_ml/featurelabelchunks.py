from usl_models.flood_ml.metastore import FirestoreDataHandler
from google.cloud.firestore_v1.document import DocumentReference

import json
import re

"""
This class is used to generate the feature, label, rainfall and temporal chunks for a given simulation name.

The class has checks for the existence of the simulation name in the Metastore.
When a simulation name is found, it is cross referanced with study_area to make sure feature chunks can be generated

"""


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DocumentReference):
            return str(obj)
        return super().default(obj)


class GenerateFeatureLabelChunks:
    def __init__(self):

        # Create an instance of FirestoreDataHandler
        self.data_handler = FirestoreDataHandler()

    def _get_study_area_from_metastore(self):
        """
        This method generates a list of study area names present in Metastore.
        """
        study_area_collection = "study_areas"
        study_area_documents = self.data_handler._find_simulation_documents(
            study_area_collection
        )
        study_area_names = []
        for study_area_document in study_area_documents:
            study_area_names.append(study_area_document.get("document_id"))
        return study_area_names

    def _get_sim_names_from_metastore(self):
        """
        This method generates a list of simulation names present in Metastore.
        """
        simulation_collection = "simulations"
        simulation_documents = self.data_handler._find_simulation_documents(
            simulation_collection
        )
        sim_names = []
        for simulation_document in simulation_documents:
            sim_names.append(simulation_document.get("document_id"))
        return sim_names

    def _get_city_cat_config_from_metastore(self):
        """
        This method generates a list of simulation names present in Metastore.
        """
        city_cat_collection = "city_cat_rainfall_configs"
        city_cat_documents = self.data_handler._find_simulation_documents(
            city_cat_collection
        )
        city_cat_names = []
        for _document in city_cat_documents:
            city_cat_names.append(_document.get("document_id"))
        return city_cat_names

    def compare_study_area_sim_chunks(self, sim_name):
        """
        This method compares the study area and simulation chunks.
        """
        # Get the study area and simulation chunks for the given simulation name
        simulation_document, configuration_ref, study_area_ref = (
            self._get_study_area_config_from_sim_name(sim_name)
        )
        label_chunks = self._get_label_chunks_for_sim(sim_name)
        label_chunks_num = len(label_chunks)

        if study_area_ref:
            study_area_chunks = self._get_feature_chunks_for_study_area(study_area_ref)
            study_area_chunks_num = len(study_area_chunks)
            print(
                f"**** Study area chunks: {study_area_chunks}, Label chunks: {label_chunks}"
            )
            try:
                if study_area_chunks_num == label_chunks_num:
                    print(
                        f"Study area and label chunks match for simulation name: {sim_name}"
                    )
                    return True

            except ValueError as e:
                print(
                    f"Study area and label chunks do not match for simulation name: {sim_name}"
                )
                print(f"Study area chunks: {study_area_chunks}")
                print(f"Label chunks: {label_chunks}")
                print(f"Error message: {e}")
                return False
        else:
            print(f"Study area for simulation name: {sim_name} not found.")
            return False

    def _get_feature_chunks_for_study_area(self, study_area_ref):
        """
        This method fetches the feature chunks for the given study area.
        """
        print(type(study_area_ref))

        # Fetch the document
        study_area_doc = study_area_ref.get()

        # Access the chunks subcollection
        try:
            if study_area_doc:
                chunks_collection_ref = study_area_ref.collection("chunks")
                chunks = [doc.to_dict() for doc in chunks_collection_ref.stream()]

                # Extract feature_matrix_path from each chunk and store in a list
                feature_matrix_paths = [chunk["feature_matrix_path"] for chunk in chunks]

                # Sort the list based on the numbers in the paths
                feature_matrix_paths.sort(
                    key=lambda path: [int(num) for num in re.findall(r"\d+", path)]
                )

                # print(f"Feature Matrix Paths: {feature_matrix_paths}")
                return feature_matrix_paths
            else:
                print("Unable to create feature matrix, check the doc!")
        except ValueError as e:
            print(f"An error occurred in creating feature matrix: {e}")

    def _get_study_area_config_from_sim_name(self, sim_name):
        """
        This method fetches the study area for a given simulation name.
        """
        # Query the "simulations" collection for the given simulation name
        simulation_document = self.data_handler._find_document_by_id(
            "simulations", sim_name
        )

        # Check if the simulation document exists and has label_chunks
        if simulation_document is not None:
            label_chunks = self._get_label_chunks_for_sim(sim_name)
            # print("Label chunks: ", label_chunks)
            if label_chunks:
                # Get configuration and study_area from the sim document
                configuration_ref = simulation_document.get("configuration")
                study_area_ref = simulation_document.get("study_area")
                # verify if study_area_ref actually exist in the "study_areas" collection
                study_area_document = self.data_handler._find_document_by_id(
                    "study_areas", study_area_ref.id
                )
                if study_area_document is not None:
                    return simulation_document, configuration_ref, study_area_ref
                else:
                    print(f"Study area not found for simulation name: {sim_name}")
                    return None, None, None

        # If no match is found, return None
        return None, None, None

    def _get_label_chunks_for_sim(self, sim_name):
        """
        This method fetches the label chunks for the given simulation name.
        """
        simulation_collection = "simulations"
        simulation_documents = self.data_handler._find_simulation_documents(
            simulation_collection
        )

        for sim_doc in simulation_documents:
            document_id = sim_doc.get("document_id")
            if document_id == sim_name:
                # Extract the label chunks
                label_chunks = sim_doc.get("subcollections", {}).get("label_chunks", [])
                label_chunks = [chunk.get("gcs_uri") for chunk in label_chunks]

                # Sort the list based on the numbers in the paths
                label_chunks.sort(
                    key=lambda path: [int(num) for num in re.findall(r"\d+", path)]
                )

                return label_chunks

        print(f"No document found for simulation name: {sim_name}")
        return []

    def _create_sim_name_dict(self, sim_name, configuration, study_area_ref):
        """
        This method creates a dictionary of simulation name, configuration and study area references.
        """
        sim_name_dict = {}
        if sim_name:
            sim_name_dict["sim_name"] = sim_name
        if configuration:
            sim_name_dict["configuration"] = configuration.get().to_dict()
        if study_area_ref:
            sim_name_dict["study_area_ref"] = study_area_ref.get().to_dict()
        return sim_name_dict

    def get_rainfall_config(self, sim_name):
        """
        This method fetches the rainfall configuration for each simulation.
        """
        #print("Generating rainfall data...")
        sim_name, configuration, study_area_ref = (
            self._get_study_area_config_from_sim_name(sim_name)
        )
        sim_name_dict = self._create_sim_name_dict(
            sim_name, configuration, study_area_ref
        )

        print("Fetching rainfall configuration...")

        try:
            if configuration:
                print("Configuration: ", configuration.id)

                city_cat_rainfall_configs = configuration.get()

                if city_cat_rainfall_configs is not None:
                    rainfall_duration = city_cat_rainfall_configs.get(
                        "rainfall_duration"
                    )
                    print("Rainfall duration: ", rainfall_duration)

                    return rainfall_duration, sim_name_dict

                else:
                    print("No document found for reference")
                    return None, sim_name_dict
        except ValueError as e:
            print(f"Error in generating temporal chunks: {e}")
            return None, sim_name_dict

        # If the method hasn't returned by this point, return None and the sim_name_dict
        return None, sim_name_dict

    def get_temporal_chunks(self, sim_name):
        """
        This method fetches the temporal chunks for a given simulation name.
        """
        print("Creating temporal chunks...")
        sim_name, configuration, study_area_ref = (
            self._get_study_area_config_from_sim_name(sim_name)
        )
        sim_name_dict = self._create_sim_name_dict(
            sim_name, configuration, study_area_ref
        )

        try:
            if configuration:
                print("Configuration: ", configuration.id)

                city_cat_rainfall_configs = self.data_handler._fetch_document(
                    configuration
                )

                if city_cat_rainfall_configs:
                    # Extract the as_vector_gcs_uri and convert to a list
                    as_vector_gcs_uri = city_cat_rainfall_configs.get(
                        "as_vector_gcs_uri"
                    )

                    return [as_vector_gcs_uri], sim_name_dict

                else:
                    print(f"No document found for reference: {configuration.id}")
                    return None, sim_name_dict
        except ValueError as e:
            print(f"Error in generating temporal chunks: {e}")
            return None, sim_name_dict

        # If the method hasn't returned by this point, return None and the sim_name_dict
        return None, sim_name_dict

    def get_label_chunks(self, sim_name_input):
        """
        This method fetches the label chunks for all simulations in metastore.
        """
        print("Creating label chunks...")
        sim_name, configuration, study_area_ref = (
            self._get_study_area_config_from_sim_name(sim_name_input)
        )
        sim_name_dict = self._create_sim_name_dict(
            sim_name, configuration, study_area_ref
        )
        try:
            if sim_name:
                return self._get_label_chunks_for_sim(sim_name_input), sim_name_dict
        except ValueError as e:
            print(f"Error in generating label chunks: {e}")

        # If the method hasn't returned by this point, return an empty list and the sim_name_dict
        return [], sim_name_dict

    def get_feature_chunks(self, sim_name):
        """
        This method fetches the feature chunks for a given simulation name.
        """
        print("Creating feature chunks...")
        sim_name, configuration, study_area_ref = (
            self._get_study_area_config_from_sim_name(sim_name)
        )
        # create a dict to store the sim_name, configuration and study_area_ref
        sim_name_dict = self._create_sim_name_dict(
            sim_name, configuration, study_area_ref
        )

        if sim_name:
            # Fetch the document
            study_area_doc = study_area_ref.get()

            # Access the chunks subcollection
            try:
                if study_area_doc.exists:
                    chunks_collection_ref = study_area_ref.collection("chunks")
                    chunks = [doc.to_dict() for doc in chunks_collection_ref.stream()]

                    # Extract feature_matrix_path from each chunk and store in a list
                    feature_matrix_paths = [
                        chunk["feature_matrix_path"] for chunk in chunks
                    ]

                    # Sort the list based on the numbers in the paths
                    feature_matrix_paths.sort(
                        key=lambda path: [int(num) for num in re.findall(r"\d+", path)]
                    )

                    # print(f"Feature Matrix Paths: {feature_matrix_paths}")
                    return feature_matrix_paths, sim_name_dict
                else:
                    print("Unable to create feature matrix, check the doc!")
            except ValueError as e:
                print(f"An error occurred in creating feature matrix: {e}")

        else:
            print("Simulation not found or it doesn't have feature chunks.")

        # If the method hasn't returned by this point, return an empty list and the sim_name_dict
        return [], sim_name_dict