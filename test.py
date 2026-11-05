import requests

url = 'http://localhost:9696/predict'

ad = {
    'title': 'Middle Python Developer',
    'description': (
        'We are looking for a Python developer to build backend services with Django '
        'and FastAPI. You will design REST APIs, work with PostgreSQL and Redis, and '
        'write tests with pytest. 3+ years of commercial experience with Python, '
        'Docker and AWS. Upper-intermediate English.'
    ),
}

response = requests.post(url, json=ad)
response.raise_for_status()
result = response.json()

print('category:', result['category'])
for category, score in result['scores'].items():
    print(f'{category:15} {score:.3f}')
