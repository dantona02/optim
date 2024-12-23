import requests
import os
from datetime import timedelta
import time

class DiscordNotifier:
    def __init__(self, webhook_url, total_steps, seq_file, n_cest_pools, n_isochromats, device="cpu"):
        """
        Initialisiert die DiscordNotifier-Klasse.

        :param webhook_url: Discord-Webhook-URL
        :param total_steps: Gesamtschritte für die Simulation
        :param seq_file: Name der Sequenzdatei
        :param n_cest_pools: Anzahl der CEST-Pools
        :param n_isochromats: Anzahl der Isochromaten
        :param device: Name des verwendeten Geräts
        """
        self.webhook_url = webhook_url.rstrip("/")
        self.total_steps = total_steps
        self.seq_filename = os.path.basename(seq_file)
        self.n_cest_pools = n_cest_pools
        self.n_isochromats = n_isochromats
        self.device = device
        self.message_id = None
        self.last_update_percentage = 0

    def _generate_bar(self, progress_percentage):
        """
        Erzeugt einen Fortschrittsbalken basierend auf dem Prozentsatz.
        """
        bar_length = 30
        filled_length = int(bar_length * progress_percentage // 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        return f"`{bar} {progress_percentage:.2f}%`"

    def send_initial_embed(self):
        """
        Sendet das initiale Embed.
        """
        embed = {
            "title": "Simulation started 🚀",
            "description": "",
            "color": 0x329fff,
            "fields": [
                {"name": "Sequence", "value": f"_{self.seq_filename}_", "inline": False},
                {"name": "N-CEST", "value": f"{self.n_cest_pools}", "inline": True},
                {"name": "N-Iso", "value": f"{self.n_isochromats}", "inline": True},
                {"name": "Simulation", "value": self._generate_bar(0), "inline": False},
                {"name": "Step", "value": f"{0}/{self.total_steps}", "inline": True},
                {"name": "Percentage", "value": f"{0:.2f}%", "inline": True}
            ],
            "footer": {
                "text": f"BMCTool | Device: {self.device}",
                "icon_url": "https://i.imgur.com/soRZIOf.png"
            }
        }
        response = requests.post(self.webhook_url + "?wait=true", json={"embeds": [embed]})
        if response.status_code == 200:
            self.message_id = response.json().get("id")
        else:
            print(f"Error sending embed: {response.text}")

    def update_progress(self, current_step):
        """
        Aktualisiert den Fortschrittsbalken basierend auf dem aktuellen Fortschritt.
        """
        if not self.message_id:
            print("No message_id stored. Cannot update embed.")
            return

        progress_percentage = (current_step / self.total_steps) * 100
        embed = {
            "title": "Simulation running 🔄",
            "description": "",
            "color": 0x329fff,
            "fields": [
                {"name": "Sequence", "value": f"_{self.seq_filename}_", "inline": False},
                {"name": "N-CEST", "value": f"{self.n_cest_pools}", "inline": True},
                {"name": "N-Iso", "value": f"{self.n_isochromats}", "inline": True},
                {"name": "Simulation", "value": self._generate_bar(progress_percentage), "inline": False},
                {"name": "Step", "value": f"{current_step}/{self.total_steps}", "inline": True},
                {"name": "Percentage", "value": f"{progress_percentage:.2f}%", "inline": True}
            ],
            "footer": {
                "text": f"BMCTool | Device: {self.device}",
                "icon_url": "https://i.imgur.com/soRZIOf.png"
            }
        }

        edit_url = f"{self.webhook_url}/messages/{self.message_id}"
        response = requests.patch(edit_url, json={"embeds": [embed]})
        if response.status_code != 200:
            print(f"Error updating embed: {response.text}")

    def send_completion_embed(self, elapsed_time):
        """
        Sendet ein Abschluss-Embed nach Beendigung der Simulation.
        """
        embed = {
            "title": "Simulation completed ✅",
            "description": "",
            "color": 0x16e819,
            "fields": [
                {"name": "Sequence", "value": f"_{self.seq_filename}_", "inline": False},
                {"name": "N-CEST", "value": f"{self.n_cest_pools}", "inline": True},
                {"name": "N-Iso", "value": f"{self.n_isochromats}", "inline": True},
                {"name": "Simulation", "value": self._generate_bar(100), "inline": False},
                {"name": "Duration", "value": f"`{str(elapsed_time)}`", "inline": True},
                {"name": "Total events", "value": f"{self.total_steps}", "inline": True}
            ],
            "footer": {
                "text": f"BMCTool | Device: {self.device}",
                "icon_url": "https://i.imgur.com/soRZIOf.png"
            }
        }

        edit_url = f"{self.webhook_url}/messages/{self.message_id}"
        response = requests.patch(edit_url, json={"embeds": [embed]})
        if response.status_code != 200:
            print(f"Error updating embed: {response.text}")
    
    def send_failed_embed(self, error_message):
        """
        Sendet ein Abschluss-Embed nach Beendigung der Simulation.
        """
        embed = {
            "title": "Simulation failed ❌",
            "description": f"Error: {str(error_message)}",
            "color": 0xff0000,
            "fields": [
                {"name": "Sequence", "value": f"_{self.seq_filename}_", "inline": False},
                {"name": "N-CEST", "value": f"{self.n_cest_pools}", "inline": True},
                {"name": "N-Iso", "value": f"{self.n_isochromats}", "inline": True}
            ],
            "footer": {
                "text": f"BMCTool | Device: {self.device}",
                "icon_url": "https://i.imgur.com/soRZIOf.png"
            }
        }

        edit_url = f"{self.webhook_url}/messages/{self.message_id}"
        response = requests.patch(edit_url, json={"embeds": [embed]})
        if response.status_code != 200:
            print(f"Error updating embed: {response.text}")