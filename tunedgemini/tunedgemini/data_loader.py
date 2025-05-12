from sklearn.datasets import fetch_20newsgroups
import email
import re
import pandas as pd


def preprocess_newsgroup_row(data):
    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate the text to fit within the input limits
    text = text[:40000]

    return text


def preprocess_newsgroup_data(newsgroup_dataset):
    # Put data points into dataframe
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
    )
    # Clean up the text
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    # Match label to target name index
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])

    return df

def preprocess_newsgroup_row(data):

    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate the text to fit within the input limits
    text = text[:40000]

    return text

def sample_data(df, num_samples, classes_to_keep):
    # Sample rows, selecting num_samples of each Label.
    df = (
        df.groupby("Label")[df.columns]
        .apply(lambda x: x.sample(num_samples))
        .reset_index(drop=True)
    )

    df = df[df["Class Name"].str.contains(classes_to_keep)]
    df["Class Name"] = df["Class Name"].astype("category")

    return df
def load_data():

    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_test = fetch_20newsgroups(subset="test")
    df_train = preprocess_newsgroup_data(newsgroups_train)
    df_test = preprocess_newsgroup_data(newsgroups_test)
    # View list of class names for dataset
    TRAIN_NUM_SAMPLES = 50
    TEST_NUM_SAMPLES = 10
    # Keep rec.* and sci.*
    CLASSES_TO_KEEP = "^rec|^sci"

    df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
    df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)
    return df_train, df_test


def sample_row():
    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_test = fetch_20newsgroups(subset="test")
    df_train, df_test = load_data(newsgroups_train, newsgroups_test)


    print(df_train.head())
    print(df_test.head())
    sample_idx = 0
    sample_row = preprocess_newsgroup_row(newsgroups_test.data[sample_idx])
    sample_label = newsgroups_test.target_names[newsgroups_test.target[sample_idx]]
    
    return df_train, df_test, sample_row, sample_label