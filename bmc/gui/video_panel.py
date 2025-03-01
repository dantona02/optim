from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QPixmap, QImage
import os
from pathlib import Path


class VideoPanel(QWidget):
    """A class representing the video panel in the BMC Simulator GUI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_video_path = None
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface for the video panel."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Create video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 480)
        
        # Create media player
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        
        # Add placeholder message when no video is loaded
        self.placeholder = QLabel("No animation has been created yet.\nCreate an animation from the control panel.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 16px;
                padding: 20px;
                background-color: #2A2A2A;
                border-radius: 10px;
            }
        """)
        
        # Add widgets to layout
        layout.addWidget(self.placeholder)
        layout.addWidget(self.video_widget)
        
        # Hide video widget initially
        self.video_widget.setVisible(False)
        
        # Media controls layout
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 10, 0, 0)
        
        # Play button
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_video)
        
        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_video)
        
        # Download button
        self.download_btn = QPushButton("Download")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self.download_video)
        
        # Add buttons to controls layout
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.download_btn)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def set_video(self, video_path):
        """Set the video to display."""
        if os.path.exists(video_path):
            self.current_video_path = video_path
            self.placeholder.setVisible(False)
            self.video_widget.setVisible(True)
            
            # Load the video
            self.media_player.setSource(QUrl.fromLocalFile(video_path))
            
            # Enable buttons
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.download_btn.setEnabled(True)
            
            # Play the video automatically
            self.play_video()
        else:
            self.clear_video()
    
    def clear_video(self):
        """Clear the current video."""
        self.media_player.stop()
        self.current_video_path = None
        self.video_widget.setVisible(False)
        self.placeholder.setVisible(True)
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
    
    def play_video(self):
        """Play the current video."""
        if self.current_video_path:
            self.media_player.play()
    
    def stop_video(self):
        """Stop the current video."""
        if self.current_video_path:
            self.media_player.stop()
    
    def download_video(self):
        """Download (save) the current video."""
        if self.current_video_path:
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Video",
                f"{Path.home()}/Downloads/{Path(self.current_video_path).name}",
                "Video Files (*.mp4)"
            )
            if save_path:
                # Make sure it has .mp4 extension
                if not save_path.lower().endswith('.mp4'):
                    save_path += '.mp4'
                
                # Copy the file
                try:
                    import shutil
                    shutil.copy2(self.current_video_path, save_path)
                except Exception as e:
                    print(f"Error saving video: {e}")