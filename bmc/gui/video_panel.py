from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QSlider, QComboBox, QToolButton,
    QFrame
)
from PyQt6.QtCore import Qt, QUrl, QTimer, QSize
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QPixmap, QImage, QIcon, QColor
import os
from pathlib import Path

class VideoPanel(QWidget):
    """A class representing the video panel in the BMC Simulator GUI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_video_path = None
        self.is_paused = False
        self.current_position = 0
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface for the video panel."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Create video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 480)
        # Add a more elegant style to the video widget
        self.video_widget.setStyleSheet("""
            QVideoWidget {
                border-radius: 10px;
                background: #1a1a1a;
            }
        """)
        
        # Create media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Connect signals
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.playbackStateChanged.connect(self.playback_state_changed)
        
        # Add placeholder message when no video is loaded
        self.placeholder = QLabel("No animation has been created yet.\nCreate an animation from the control panel.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("""
            QLabel {
                color: #a0a0a0;
                font-size: 16px;
                padding: 40px;
                background-color: #222222;
                border-radius: 10px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }
        """)
        
        # Add widgets to layout
        layout.addWidget(self.placeholder)
        layout.addWidget(self.video_widget)
        
        # Hide video widget initially
        self.video_widget.setVisible(False)
        
        # Create a container for playback controls with a modern look
        controls_container = QFrame()
        controls_container.setObjectName("videoControlsContainer")
        controls_container.setStyleSheet("""
            #videoControlsContainer {
                background-color: rgba(25, 25, 25, 0.85);
                border-radius: 10px;
                padding: 0px;
                border: 1px solid rgba(60, 60, 60, 0.3);
            }
        """)
        
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setSpacing(10)
        
        # Position slider with modern styling
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.setEnabled(False)
        self.position_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #444444;
                border-radius: 2px;
            }
            
            QSlider::handle:horizontal {
                background: #4287f5;
                border: none;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #5a9bff;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            
            QSlider::sub-page:horizontal {
                background: #4287f5;
                border-radius: 2px;
            }
        """)
        controls_layout.addWidget(self.position_slider)
        
        # Time and controls in one row
        time_controls_layout = QHBoxLayout()
        time_controls_layout.setContentsMargins(0, 0, 0, 0)
        time_controls_layout.setSpacing(15)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setStyleSheet("color: white; font-size: 13px; font-weight: 500;")
        time_controls_layout.addWidget(self.time_label)
        
        time_controls_layout.addStretch(1)
        
        # Media controls in center - use a nested layout
        media_controls_layout = QHBoxLayout()
        media_controls_layout.setSpacing(30)  # Increased spacing between controls
        
        # Function to create icon buttons with custom colors
        def create_svg_icon_button(icon_path, icon_size, tooltip, callback):
            button = QToolButton()
            # Set up icon with white color
            icon = self.create_white_icon(icon_path)
            button.setIcon(icon)
            button.setIconSize(QSize(*icon_size))
            button.setToolTip(tooltip)
            button.setEnabled(False)
            button.setStyleSheet("""
                QToolButton {
                    border: none;
                    background: transparent;
                    padding: 8px;
                }
                QToolButton:hover {
                    background: rgba(255, 255, 255, 0.15);
                    border-radius: 20px;
                }
                QToolButton:pressed {
                    background: rgba(255, 255, 255, 0.25);
                }
                QToolButton:disabled {
                    opacity: 0.4;
                }
            """)
            button.clicked.connect(callback)
            return button
        
        # Rewind button
        self.rewind_btn = create_svg_icon_button(
            os.path.join(os.path.dirname(__file__), "images/backward.svg"),
            (26, 26),  # Slightly larger icons
            "Rewind 5 seconds",
            self.rewind_video
        )
        media_controls_layout.addWidget(self.rewind_btn)
        
        # Play button
        self.play_btn = create_svg_icon_button(
            os.path.join(os.path.dirname(__file__), "images/play.svg"),
            (36, 36),  # Larger play button
            "Play",
            self.play_pause_video
        )
        self.play_btn.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                padding: 10px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.15);
                border-radius: 25px;
            }
            QToolButton:pressed {
                background: rgba(255, 255, 255, 0.25);
            }
            QToolButton:disabled {
                opacity: 0.4;
            }
        """)
        media_controls_layout.addWidget(self.play_btn)
        
        # Forward button
        self.forward_btn = create_svg_icon_button(
            os.path.join(os.path.dirname(__file__), "images/forward.svg"),
            (26, 26),  # Slightly larger icons
            "Forward 5 seconds",
            self.forward_video
        )
        media_controls_layout.addWidget(self.forward_btn)
        
        time_controls_layout.addLayout(media_controls_layout)
        time_controls_layout.addStretch(1)
        
        # Right controls group
        right_controls_layout = QHBoxLayout()
        right_controls_layout.setSpacing(15)
        
        # Speed control with modern styling
        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(6)
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: white; font-size: 13px;")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(2)  # 1.0x default
        self.speed_combo.setEnabled(False)
        self.speed_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(60, 60, 60, 0.7);
                color: white;
                border: 1px solid rgba(80, 80, 80, 0.5);
                border-radius: 5px;
                padding: 4px 10px;
                min-width: 75px;
                font-weight: 500;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 10px;
                height: 10px;
                background: rgba(255, 255, 255, 0.7);
                border-radius: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: white;
                selection-background-color: #4287f5;
                border: 1px solid #555555;
                border-radius: 0px;
            }
        """)
        self.speed_combo.currentIndexChanged.connect(self.speed_changed)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_combo)
        right_controls_layout.addLayout(speed_layout)
        
        # Download button
        self.download_btn = create_svg_icon_button(
            os.path.join(os.path.dirname(__file__), "images/download.svg"),
            (22, 22),
            "Download Video",
            self.download_video
        )
        right_controls_layout.addWidget(self.download_btn)
        
        time_controls_layout.addLayout(right_controls_layout)
        
        # Add sub-layouts to the main controls layout
        controls_layout.addLayout(time_controls_layout)
        
        # Add controls container to main layout
        layout.addWidget(controls_container)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Set overall widget style
        self.setStyleSheet("""
            QToolTip {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
        """)
    
    def create_white_icon(self, icon_path):
        """Create white colored SVG icon from the original SVG file."""
        if icon_path.lower().endswith('.svg'):
            # Create a QIcon with white color filter
            icon = QIcon(icon_path)
            # Apply white color mode to all icon states
            icon.setIsMask(True)  # This makes the icon use the foreground color
            return icon
        else:
            # For non-SVG files just return the regular icon
            return QIcon(icon_path)
    
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
            self.rewind_btn.setEnabled(True)
            self.forward_btn.setEnabled(True)
            self.speed_combo.setEnabled(True)
            self.position_slider.setEnabled(True)
            self.download_btn.setEnabled(True)
            self.is_paused = False
            
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
        self.rewind_btn.setEnabled(False)
        self.forward_btn.setEnabled(False)
        self.speed_combo.setEnabled(False)
        self.position_slider.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.position_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
    
    def play_video(self):
        """Play the current video."""
        if self.current_video_path:
            self.media_player.play()
            pause_icon = self.create_white_icon(os.path.join(os.path.dirname(__file__), "images/pause.svg"))
            self.play_btn.setIcon(pause_icon)
            self.play_btn.setToolTip("Pause")
            self.is_paused = False
    
    def pause_video(self):
        """Pause the current video."""
        if self.current_video_path:
            self.media_player.pause()
            play_icon = self.create_white_icon(os.path.join(os.path.dirname(__file__), "images/play.svg"))
            self.play_btn.setIcon(play_icon)
            self.play_btn.setToolTip("Play")
            self.is_paused = True
            self.current_position = self.media_player.position()
    
    def play_pause_video(self):
        """Toggle between play and pause."""
        if self.is_paused:
            self.play_video()
        else:
            self.pause_video()
    
    def rewind_video(self):
        """Rewind the video by 5 seconds."""
        if self.current_video_path:
            new_position = max(0, self.media_player.position() - 5000)
            self.media_player.setPosition(new_position)
    
    def forward_video(self):
        """Forward the video by 5 seconds."""
        if self.current_video_path:
            new_position = min(self.media_player.duration(), self.media_player.position() + 5000)
            self.media_player.setPosition(new_position)
    
    def speed_changed(self, index):
        """Change the playback speed."""
        speed_options = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        if self.current_video_path and index >= 0 and index < len(speed_options):
            self.media_player.setPlaybackRate(speed_options[index])
    
    def position_changed(self, position):
        """Handle the position change in the video."""
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        
        # Update time label
        current_time = self.format_time(position)
        total_time = self.format_time(self.media_player.duration())
        self.time_label.setText(f"{current_time} / {total_time}")
    
    def duration_changed(self, duration):
        """Handle the duration change of the loaded video."""
        self.position_slider.setRange(0, duration)
        total_time = self.format_time(duration)
        self.time_label.setText(f"00:00 / {total_time}")
    
    def set_position(self, position):
        """Set the position in the video based on slider movement."""
        self.media_player.setPosition(position)
    
    def playback_state_changed(self, state):
        """Handle playback state changes."""
        if state == QMediaPlayer.PlaybackState.StoppedState:
            # Video reached the end or was stopped
            # Rewind to beginning
            self.media_player.setPosition(0)
            play_icon = self.create_white_icon(os.path.join(os.path.dirname(__file__), "images/play.svg"))
            self.play_btn.setIcon(play_icon)
            self.play_btn.setToolTip("Play")
            self.is_paused = True
    
    def format_time(self, milliseconds):
        """Format time in milliseconds to MM:SS format."""
        seconds = int(milliseconds / 1000)
        minutes = int(seconds / 60)
        seconds %= 60
        return f"{minutes:02d}:{seconds:02d}"
    
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