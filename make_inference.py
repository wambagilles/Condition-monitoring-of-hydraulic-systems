import requests

def make_inference(cycle_id):
    # URL for the prediction endpoint
    url = 'http://localhost:9696/predict'


    # Customer data for making predictions
    cycle = {
        "cycle_id": int(cycle_id),

    }

    # Make a POST request to the prediction endpoint with customer data
    response = requests.post(url, json=cycle).json()

    # Print the prediction response
    print(response)

    # Check if cycle is optimal or not and True and take action
    if response['condition'] == "Optimal":
        print(f"The cycle {cycle['cycle_id']} is Optimal" )
    else:
        print(f"The cycle {cycle['cycle_id']} is NOT Optimal" )
    
    return response



