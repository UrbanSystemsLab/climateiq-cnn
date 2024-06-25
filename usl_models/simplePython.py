import numpy as np
# from usl_models.flood_ml.metastore import FirestoreDataHandler
# from google.cloud import firestore
# from usl_models.flood_ml.settings import Settings
# from usl_models.flood_ml.featurelabelchunks import GenerateFeatureLabelChunks


# Load the numpy file
data = np.load('/Users/pskulkarni/Downloads/Manhattan_config_v1_Rainfall_Data_2.txt_0_0.npy')

# Print the shape of the loaded data
print(data.shape)
# firestore_client: firestore.Client = None
# settings: Settings = None
# ms = FirestoreDataHandler(firestore_client, settings)
# class_label = GenerateFeatureLabelChunks()
# sim_names = class_label._get_sim_names_from_metastore()
# print(sim_names)
# simulation_documents = ms._find_simulation_documents("simulations")
# study_area_documents = ms._find_simulation_documents("study_areas")

# for sim_doc in simulation_documents:
#     print(sim_doc.get("label_chunks"))

# # for study_area_doc in study_area_documents:
# #     print(study_area_doc.to_dict())
