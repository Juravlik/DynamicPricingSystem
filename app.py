import requests
import pandas as pd
import numpy as np
import uuid
from typing import List, Optional, Callable
import time


class AppHandler:

    def __init__(self,
                 uuid: str,
                 url_begin: str,
                 url_task_data_get: str,
                 url_task_result_post: str,
                 url_task_result_get: str
                 ):

        self.uuid = uuid
        self.url_begin = url_begin
        self.url_task_data_get = url_task_data_get
        self.url_task_result_post = url_task_result_post
        self.url_task_result_get = url_task_result_get

    def start(self):
        req = requests.post(self.url_begin.format(uuid=self.uuid))

        print('Start working:')
        print(req.json())

    def get_batch(self) -> Optional[pd.DataFrame]:
        response = requests.get(self.url_task_data_get.format(uuid=self.uuid))

        if response.json()['status'] == 'batch processing finished ':
            print('Batches are over')
            return None

        return pd.read_json(response.json())

    def send_batch_predictions(self, batch_df: pd.DataFrame):
        requests.post(self.url_task_result_post.format(uuid=self.uuid),
                      data=batch_df.to_json(orient='records'))

    def get_batch_results(self) -> pd.DataFrame:
        response = requests.get(self.url_task_result_get.format(uuid=self.uuid))

        return pd.read_json(response.json())





if __name__ == "__main__":

    UUID = uuid.uuid4().hex
    print(UUID)

    URL_BEGIN_DATA = 'https://lab.karpov.courses/hardml-api/project-1/task/{uuid}/begin'
    req = requests.post(URL_BEGIN_DATA.format(uuid=UUID))

    print(req.json())
