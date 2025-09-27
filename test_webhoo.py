import requests

# # Replace with your Railway webhook service URL
# url = "https://scoresignalbot.up.railway.app/webhook"

# # Send empty POST (should return "no message" not 404)
# response = requests.post(url, json={})
# print(response.status_code, response.json())
requests.post("https://scoresignalbot.up.railway.app/webhook", json={})