import json
import os
import onnxruntime

def init():
    global sess
    sess = onnxruntime.InferenceSession(
        os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.onnx")
    )


def run(request):
    print(request)
    request_data = json.loads(request)

    # Run inference
    test = sess.run(
        {
            'gender': 'Male',
            'age': 34,
            'hypertension': 0,
            'heart_disease': 1,
            'ever_married': 'Yes',
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'avg_glucose_level': 219.34,
            'bmi': 34.4,
            'smoking_status': 'formerly smoked',
        },
    )

    return test
