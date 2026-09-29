import torch
import torch.nn as nn
from transformers import ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

MODEL_NAME = "abhilash88/age-gender-prediction"

_MODEL = None
_PROCESSOR = None


class AgeGenderViTModel(ViTPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.vit = ViTModel(
            config,
            add_pooling_layer=False
        )

        self.age_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.num_labels = 2
        self.config.num_labels = 2
        self.classifier = nn.Linear(
            config.hidden_size,
            2
        )

        self.post_init()

    def forward(
        self,
        pixel_values=None,
        labels=None,
        **kwargs
    ):
        outputs = self.vit(
            pixel_values=pixel_values,
            **kwargs
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]

        age_output = self.age_head(
            pooled_output
        )

        gender_output = self.gender_head(
            pooled_output
        )

        logits = torch.cat(
            [age_output, gender_output],
            dim=1
        )

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else None,
            attentions=outputs.attentions
            if hasattr(outputs, "attentions")
            else None
        )


def _load_model_once():

    global _MODEL
    global _PROCESSOR

    if _MODEL is None:

        from transformers import (
            AutoConfig,
            AutoImageProcessor
        )

        print("Loading Age/Gender model...")

        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        _MODEL = AgeGenderViTModel.from_pretrained(
            MODEL_NAME,
            config=config,
            trust_remote_code=True
        )

        _MODEL.eval()

        _PROCESSOR = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )

        print("Age/Gender model ready.")

    return _MODEL, _PROCESSOR


def predict_age_gender(image):

    model, processor = _load_model_once()

    inputs = processor(
        images=image,
        return_tensors="pt",
        do_center_crop=False
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    logits = outputs.logits[0]

    age = int(
        round(
            logits[0].item()
        )
    )

    age = max(
        0,
        min(100, age)
    )

    female_probability = float(
        logits[1].item()
    )

    male_probability = (
        1.0 - female_probability
    )

    if female_probability >= 0.5:
        gender = "Female"
    else:
        gender = "Male"

    confidence = max(
        female_probability,
        male_probability
    )

    return {
        "age": age,
        "gender": gender,
        "gender_confidence": confidence,
        "gender_probability_male": male_probability,
        "gender_probability_female": female_probability,
        "label": f"{age} years, {gender}",
        "score": confidence
    }


def simple_predict(image):

    result = predict_age_gender(image)

    return (
        f"{result['age']} years, "
        f"{result['gender']} "
        f"({result['gender_confidence']:.1%} confidence)"
    )
