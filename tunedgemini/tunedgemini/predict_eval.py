from tqdm.rich import tqdm as tqdmr
import tqdm
import warnings
from google.api_core import retry

from tunedgemini.data_loader import sample_row, sample_data, load_data
from google import genai
from google.genai import types
from google.api_core import retry

system_instruct = """
You are a classification service. You will be passed input that represents
a newsgroup post and you must respond with the newsgroup from which the post
originates.
"""
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
@retry.Retry(predicate=is_retriable)
def predict_label(post: str, client, model_id) -> str:
    response = client.models.generate_content(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=system_instruct),
        contents=post)

    rc = response.candidates[0]

    # Any errors, filters, recitation, etc we can mark as a general error
    if rc.finish_reason.name != "STOP":
        return "(error)"
    else:
        # Clean up the response.
        return response.text.strip()        
def eval_model(client, df_test, model_id):

    
    tqdmr.pandas()

    # But suppress the experimental warning
    warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)


    # Further sample the test data to be mindful of the free-tier quota.
    df_baseline_eval = sample_data(df_test, 2, '.*')

    # Make predictions using the sampled data.
    df_baseline_eval['Prediction'] = df_baseline_eval['Text'].progress_apply(lambda text: predict_label(text, client, model_id))

    # And calculate the accuracy.
    accuracy = (df_baseline_eval["Class Name"] == df_baseline_eval["Prediction"]).sum() / len(df_baseline_eval)
    print(df_baseline_eval.head())
    print(f"Accuracy: {accuracy:.2%}")
    return df_baseline_eval


@retry.Retry(predicate=is_retriable)
def classify_text(client, text: str, model_id: str) -> str:
    """Classify the provided text into a known newsgroup."""
    response = client.models.generate_content(
        model=model_id, contents=text)
    rc = response.candidates[0]

    # Any errors, filters, recitation, etc we can mark as a general error
    if rc.finish_reason.name != "STOP":
        return "(error)"
    else:
        return rc.content.parts[0].text


def eval_tuned_model(client, df_test, model_id):

    # The sampling here is just to minimise your quota usage. If you can, you should
    # evaluate the whole test set with `df_model_eval = df_test.copy()`.


    df_model_eval = sample_data(df_test, 4, '.*')

    df_model_eval["Prediction"] = df_model_eval["Text"].progress_apply(lambda text: classify_text(client, text, model_id))

    accuracy = (df_model_eval["Class Name"] == df_model_eval["Prediction"]).sum() / len(df_model_eval)
    print(f"Accuracy: {accuracy:.2%}")