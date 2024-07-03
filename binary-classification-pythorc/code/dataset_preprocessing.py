import pandas as pd


def get_dataset(dataset_type):
    # Load datasets
    print('Loading data ...')
    df = pd.read_csv('../dataset/edos_labelled_aggregated.csv')
    print('Data loaded --> dataset has ' + str(df.shape[0]) + ' rows.')

    # Cleaning dataset
    print('Cleaning data ...')
    columns_to_drop = ['label_category', 'label_vector']
    df = df.drop(columns=columns_to_drop)
    df = df.rename(columns={'text': 'sentence', 'label_sexist': 'label'})
    print('Data cleaned...')

    df = df[df['split'] == dataset_type]
    counts = df['label'].value_counts()

    print(dataset_type + ' set has ' + str(df.shape[0]) + ' rows: ')
    print(counts)

    return df


if __name__ == '__main__':
    print(get_dataset('train').shape[0] / 20000 * 100)
    print(get_dataset('test').shape[0] / 20000 * 100)
    print(get_dataset('dev').shape[0] / 20000 * 100)