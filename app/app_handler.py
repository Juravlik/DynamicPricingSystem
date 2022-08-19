import requests
import pandas as pd
import numpy as np
import uuid
from typing import List, Optional, Callable
from utils.utils import wait


class AppHandler:

    def __init__(self,
                 uuid: str,
                 url_begin: str,
                 url_task_data_get: str,
                 url_task_result_post: str,
                 url_task_result_get: str,
                 url_task_score_get: str):

        self.uuid = uuid
        self.url_begin = url_begin
        self.url_task_data_get = url_task_data_get
        self.url_task_result_post = url_task_result_post
        self.url_task_result_get = url_task_result_get
        self.url_task_score_get = url_task_score_get

    def start(self):
        req = requests.post(self.url_begin.format(uuid=self.uuid))

        print('Start working:')
        print(req.json())

    def get_batch(self) -> Optional[pd.DataFrame]:
        response = requests.get(self.url_task_data_get.format(uuid=self.uuid))

        if 'status' in response.json() and response.json()['status'] == 'batch processing finished ':
            print('Batches are over')
            return None

        return pd.read_json(response.json())

    def send_batch_predictions(self, batch_df: pd.DataFrame):
        requests.post(self.url_task_result_post.format(uuid=self.uuid),
                      data=batch_df.to_json(orient='records'))

    def get_batch_results(self) -> pd.DataFrame:
        response = requests.get(self.url_task_result_get.format(uuid=self.uuid))

        return pd.read_json(response.json())

    def get_current_score(self) -> Optional[float]:
        response = requests.get(self.url_task_score_get.format(uuid=self.uuid))

        if 'result' in response.json():
            return response.json()['result']
        else:
            return None


if __name__ == "__main__":

    UUID = uuid.uuid4().hex
    print(UUID)

    app_handler = AppHandler(uuid=UUID,
                             url_begin='https://lab.karpov.courses/hardml-api/project-1/task/{}/begin'.format(UUID),
                             url_task_data_get='https://lab.karpov.courses/hardml-api/project-1/task/{}/data'\
                             .format(UUID),
                             url_task_result_get='https://lab.karpov.courses/hardml-api/project-1/task/{}/result'\
                             .format(UUID),
                             url_task_result_post='https://lab.karpov.courses/hardml-api/project-1/task/{}/result/'\
                             .format(UUID),
                             url_task_score_get='https://lab.karpov.courses/hardml-api/project-1/task/{}/lms_result'\
                             .format(UUID))

    app_handler.start()

    df_batch = app_handler.get_batch()

    print(df_batch.head())
    print(df_batch.tail())



