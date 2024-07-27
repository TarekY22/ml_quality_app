from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import torch

# Import libraries for robustness, correctness, and fairness
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from cleverhans.attacks import fast_gradient_method
from secml.adv.attacks import CAttackEvasionPGDLS
from foolbox import PyTorchModel, accuracy, samples
from armory.utils.config_loading import load_model
import textattack
import artemis
import deepcheck
import cerberus
import tensorflow_model_analysis as tfma
import great_expectations as ge
from aif360.metrics import BinaryLabelDatasetMetric
from fairlearn.metrics import demographic_parity_difference
from themis_ml import fairness
from fairness_indicators import evaluate_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def load_model_from_file(file, framework):
    if framework == 'scikit-learn':
        return joblib.load(file)
    elif framework == 'tensorflow':
        return tf.keras.models.load_model(file)
    elif framework == 'pytorch':
        return torch.load(file)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def load_data_from_file(file):
    return pd.read_csv(file)

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    model_file = request.files['model']
    data_file = request.files['data']
    framework = request.form['framework']
    model_type = request.form['model_type']
    evaluation_type = request.form['evaluation_type']

    model = load_model_from_file(model_file, framework)
    data = load_data_from_file(data_file)
    
    x_train = data.drop(columns=['label'])
    y_train = data['label']

    if evaluation_type == 'robustness':
        if framework == 'scikit-learn':
            classifier = SklearnClassifier(model=model)
        elif framework == 'tensorflow':
            classifier = tf.keras.models.Model(model)
        elif framework == 'pytorch':
            classifier = PyTorchModel(model=model)
        attack = FastGradientMethod(estimator=classifier)
        x_test_adv = attack.generate(x=x_train)
        accuracy = np.sum(np.argmax(classifier.predict(x_test_adv), axis=1) == y_train) / len(y_train)
        
        result = {
            "evaluation_type": evaluation_type,
            "metrics": {
                "adversarial_accuracy": accuracy
            }
        }

    elif evaluation_type == 'correctness':
        eval_result = tfma.EvalResult()
        metrics = tfma.view.render_slicing_metrics(eval_result)
        
        result = {
            "evaluation_type": evaluation_type,
            "metrics": metrics
        }

    elif evaluation_type == 'fairness':
        dataset = BinaryLabelDataset(df=data)
        metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{'sex': 1}], unprivileged_groups=[{'sex': 0}])
        disparate_impact = metric.disparate_impact()

        result = {
            "evaluation_type": evaluation_type,
            "metrics": {
                "disparate_impact": disparate_impact
            }
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
