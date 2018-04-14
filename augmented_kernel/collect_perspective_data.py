from googleapiclient import discovery
from api_keys import perspective_api_key
import json

API_KEY = perspective_api_key

def get_perspective_toxicity(comment):
	# Generates API client object dynamically based on service name and version.
	service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)

	analyze_request = {
	  'comment': { 'text': comment },
	  'requestedAttributes': {'TOXICITY': {}}
	}

	response = service.comments().analyze(body=analyze_request).execute()

	# print(json.dumps(response, indent=2))
	return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]



# print(get_perspective_toxicity("You are a terrible, awful guy!"))