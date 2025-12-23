import requests
import os

# Start a session
session = requests.Session()

# Login with demo user
login_data = {
    'email': 'demo@example.com',
    'password': 'demo123'
}
response = session.post('http://127.0.0.1:5000/login', data=login_data)

if response.status_code == 200:
    print("Login successful")
else:
    print(f"Login failed: {response.status_code}")
    exit(1)

# Now test predict with a test image
if os.path.exists('test_image.png'):
    with open('test_image.png', 'rb') as f:
        files = {'file': f}
        response = session.post('http://127.0.0.1:5000/predict', files=files)

    print(f"Predict response status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if 'success' in data and data['success']:
            print("Prediction successful!")
            print(f"Top prediction: {data['top_prediction']['class']} with {data['top_prediction']['confidence']}% confidence")
        else:
            print(f"Prediction failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"Predict failed with status {response.status_code}: {response.text}")
else:
    print("Test image not found")
