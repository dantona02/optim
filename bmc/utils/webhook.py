import requests

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.message_id = None

    def send_initial_message(self, content):
        response = requests.post(self.webhook_url + "?wait=true", json={"content": content})
        if response.status_code == 200:
            self.message_id = response.json()["id"]
        else:
            print(f"Error sending message: {response.text}")

    def update_message(self, content):
        if not self.message_id:
            print("No message_id stored. Cannot update message.")
            return

        edit_url = f"{self.webhook_url}/messages/{self.message_id}"
        response = requests.patch(edit_url, json={"content": content})
        if response.status_code != 200:
            print(f"Error updating message: {response.text}")
