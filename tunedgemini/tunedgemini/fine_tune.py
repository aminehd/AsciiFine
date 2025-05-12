from collections.abc import Iterable
import random
from google.genai import types
import datetime
import time

def fine_tune(client,  df_train, base_model="models/gemini-1.5-flash-001-tuning", model_id=None):
    # Convert the data frame into a dataset suitable for tuning.
    input_data = {'examples': 
        df_train[['Text', 'Class Name']]
        .rename(columns={'Text': 'textInput', 'Class Name': 'output'})
        .to_dict(orient='records')
    }

    # If you are re-running this lab, add your model_id here.

    # Or try and find a recent tuning job.
    if not model_id:
        queued_model = None
    # Newest models first.
        for m in reversed(client.tunings.list()):
            # Only look at newsgroup classification models.
            if m.name.startswith('tunedModels/newsgroup-classification-model'):
            # If there is a completed model, use the first (newest) one.
                if m.state.name == 'JOB_STATE_SUCCEEDED':
                    model_id = m.name
                    print('Found existing tuned model to reuse.')
                    break

            elif m.state.name == 'JOB_STATE_RUNNING' and not queued_model:
                # If there's a model still queued, remember the most recent one.
                queued_model = m.name
        else:
            if queued_model:
                model_id = queued_model
            print('Found queued model, still waiting.')


        # Upload the training data and queue the tuning job.
        if not model_id:
            tuning_op = client.tunings.tune(
                base_model=f"models/{base_model}-tuning",
                training_dataset=input_data,
                config=types.CreateTuningJobConfig(
                    tuned_model_display_name="Newsgroup classification model",
                    batch_size=16,
                    epoch_count=2,
                ),
            )

            print(tuning_op.state)
            model_id = tuning_op.name

    return model_id

def get_tuned_model(client, model_id):
    MAX_WAIT = datetime.timedelta(minutes=10)

    while not (tuned_model := client.tunings.get(name=model_id)).has_ended:

        print(tuned_model.state)
        time.sleep(60)

        # Don't wait too long. Use a public model if this is going to take a while.
        if datetime.datetime.now(datetime.timezone.utc) - tuned_model.create_time > MAX_WAIT:
            print("Taking a shortcut, using a previously prepared model.")
            model_id = "tunedModels/newsgroup-classification-model-ltenbi1b"
            tuned_model = client.tunings.get(name=model_id)
            break


    print(f"Done! The model state is: {tuned_model.state.name}")

    if not tuned_model.has_succeeded and tuned_model.error:
        print("Error:", tuned_model.error)

    return tuned_model
