import pandas as pd
import os
from pydantic import BaseModel


class Feedback(BaseModel):
    sentence: str
    label: str


def add_feedback_to_dataset(feedback: Feedback):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'static', 'dataset', 'dataset.csv')

    df = pd.read_csv(dataset_path)

    # New row to append
    new_row = {'sentence': feedback.sentence, 'label': feedback.label}
    print(new_row)

    # Add the new row using .loc[]
    df.loc[len(df)] = new_row

    df.to_csv(dataset_path, index=False)
