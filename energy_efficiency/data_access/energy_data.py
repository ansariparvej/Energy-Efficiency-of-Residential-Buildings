import sys
from typing import Optional
import numpy as np
import pandas as pd
import json
from energy_efficiency.configuration.mongo_db_connection import MongoDBClient
from energy_efficiency.constant.database import DATABASE_NAME  # energy_saving
from energy_efficiency.exception import EnergyException


class EnergyData:
    """
    This class helps  to import mongodb record as csv format and export entire mongodb record as pandas dataframe
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise EnergyException(e, sys)

    def save_csv_file(self, file_path, collection_name: str, database_name: Optional[str] = None):
        try:
            """
            save (import from PC) .csv file as json format to mongodb:
            returns none
            """
            data_frame = pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]  # energy_efficiency
            else:
                collection = self.mongo_client[database_name][collection_name]  # db = energy_saving, collection = energy_efficiency
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise EnergyException(e, sys)


    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            """
            export entire collection as dataframe from mongodb:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]  # energy_efficiency
            else:
                collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise EnergyException(e, sys)
