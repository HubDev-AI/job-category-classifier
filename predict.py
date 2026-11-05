import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MAX_TITLE_LENGTH = 300
MAX_DESCRIPTION_LENGTH = 20_000


class JobAd(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_LENGTH)


class PredictResponse(BaseModel):
    category: str
    scores: dict[str, float]


app = FastAPI(
    title='job-category-classifier',
    description='Suggests one of six job-board categories (JavaScript, Java, DevOps, '
                'QA Automation, Python, Data Science) for an English job ad.',
)

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(ad):
    proba = pipeline.predict_proba([ad['title'] + ' ' + ad['description']])[0]
    scores = {c: float(p) for c, p in zip(pipeline.classes_, proba)}
    return str(pipeline.classes_[proba.argmax()]), scores


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(ad: JobAd) -> PredictResponse:
    category, scores = predict_single(ad.model_dump())
    return PredictResponse(category=category, scores=scores)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)
