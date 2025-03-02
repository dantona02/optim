from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QSlider, QComboBox, QToolButton,
    QFrame, QListView
)
from PyQt6.QtCore import Qt, QUrl, QTimer, QSize
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QPixmap, QImage, QIcon, QColor, QStandardItemModel, QStandardItem
import os
from pathlib import Path

class CustomSpeedComboBox(QComboBox):
    """Eine Custom ComboBox, die im geschlossenen Zustand keinen Text anzeigt."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Erstelle ein custom Model für die Items
        self._model = QStandardItemModel()
        self.setModel(self._model)
        
        # Speichere die original Texte
        self._texts = []
        
        # Setze eine custom View
        self._list_view = QListView()
        self.setView(self._list_view)
        
        # Style
        self.setStyleSheet("""
            QComboBox {
                background-color: transparent;
                border: none;
                border-radius: 0px;
                padding: 3px;
                min-width: 28px;
                max-width: 55px;
                margin-left: 6px;
            }
            QComboBox:hover {
                background: rgba(255, 255, 255, 0.15);
                border-radius: 10px;
            }
            QComboBox:pressed {
                background: rgba(255, 255, 255, 0.25);
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
                subcontrol-origin: padding;
                subcontrol-position: center;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(30, 30, 30, 0.95);
                color: white;
                selection-background-color: rgba(66, 135, 245, 0.5);
                selection-color: white;
                border: none;
                outline: none;
                padding: 3px;
                min-width: 70px;
            }
            QComboBox QAbstractItemView::item {
                color: white;
                min-height: 22px;
                padding: 2px 6px;
            }
        """)
    
    def addItems(self, texts):
        """Überschreibe addItems um die Texte zu speichern."""
        self._texts = texts
        # Füge leere Items hinzu für den geschlossenen Zustand
        for _ in texts:
            item = QStandardItem("")
            self._model.appendRow(item)
    
    def showPopup(self):
        """Zeige die Texte wenn das Popup geöffnet wird."""
        # Setze die Texte für alle Items
        for i, text in enumerate(self._texts):
            self._model.item(i).setText(text)
        super().showPopup()
    
    def hidePopup(self):
        """Verstecke die Texte wenn das Popup geschlossen wird."""
        super().hidePopup()
        # Leere die Texte für alle Items
        for i in range(self._model.rowCount()):
            self._model.item(i).setText("")
            
    def setEnabled(self, enabled):
        """Überschreibe setEnabled um das Styling des Dropdownpfeils zu aktualisieren."""
        super().setEnabled(enabled)
        # Manuelle Aktualisierung des Styles
        self.style().unpolish(self)
        self.style().polish(self)

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
            QSlider {
                background: transparent;
                padding: 12px 0px;
                opacity: 1;
            }
            
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(68, 68, 68, 0.7);
                border-radius: 2px;
            }
            
            QSlider::handle:horizontal {
                background: #2962FF;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #4287f5;
            }
            
            QSlider::sub-page:horizontal {
                background: #2962FF;
                border-radius: 2px;
            }

            QSlider:disabled {
                opacity: 0.35;
            }

            QSlider::handle:horizontal:disabled {
                background: #666666;
            }

            QSlider::sub-page:horizontal:disabled {
                background: #666666;
            }

            QSlider::groove:horizontal:disabled {
                background: rgba(68, 68, 68, 0.4);
            }
        """)
        controls_layout.addWidget(self.position_slider)
        
        # Time and controls in one row
        time_controls_layout = QHBoxLayout()
        time_controls_layout.setContentsMargins(0, 0, 0, 0)
        time_controls_layout.setSpacing(15)
        
        # Time display - Hintergrund transparent
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 13px;
                font-weight: 500;
                background: transparent;
            }
            QLabel[enabled="false"] {
                color: rgba(255, 255, 255, 0.4);
            }
        """)
        self.time_label.setProperty("enabled", False)
        time_controls_layout.addWidget(self.time_label)
        
        time_controls_layout.addStretch(1)
        
        # Media controls in center - use a nested layout
        media_controls_layout = QHBoxLayout()
        media_controls_layout.setSpacing(1)  # Etwas verringerten Abstand zwischen Kontrollelementen
        
        # Function to create icon buttons with custom colors
        def create_svg_icon_button(icon_path, icon_size, tooltip, callback, is_play_button=False):
            button = QToolButton()
            # Set up icon with white color
            icon = self.create_white_icon(icon_path)
            button.setIcon(icon)
            button.setIconSize(QSize(*icon_size))
            button.setToolTip(tooltip)
            button.setEnabled(False)
            
            # Angepasstes Styling mit kleinerem Hover-Radius
            if is_play_button:
                button.setStyleSheet("""
                    QToolButton {
                        border: none;
                        background: transparent;
                        padding: 6px;
                    }
                    QToolButton:hover {
                        background: rgba(255, 255, 255, 0.15);
                        border-radius: 12px;
                    }
                    QToolButton:pressed {
                        background: rgba(255, 255, 255, 0.25);
                    }
                    QToolButton:disabled {
                        opacity: 0.4;
                    }
                """)
            else:
                button.setStyleSheet("""
                    QToolButton {
                        border: none;
                        background: transparent;
                        padding: 6px;
                    }
                    QToolButton:hover {
                        background: rgba(255, 255, 255, 0.15);
                        border-radius: 10px;
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
            (22, 22),  # Etwas kleinere Icons
            "Rewind 5 seconds",
            self.rewind_video
        )
        media_controls_layout.addWidget(self.rewind_btn)
        
        # Play button - kleiner gemacht
        self.play_btn = create_svg_icon_button(
            os.path.join(os.path.dirname(__file__), "images/play.svg"),
            (22, 22),  # Kleinerer Play-Button
            "Play",
            self.play_pause_video,
            is_play_button=True
        )
        media_controls_layout.addWidget(self.play_btn)
        
        # Forward button
        self.forward_btn = create_svg_icon_button(
            os.path.join(os.path.dirname(__file__), "images/forward.svg"),
            (22, 22),  # Etwas kleinere Icons
            "Forward 5 seconds",
            self.forward_video
        )
        media_controls_layout.addWidget(self.forward_btn)
        
        # Speed control mit CustomSpeedComboBox
        self.speed_combo = CustomSpeedComboBox()
        self.speed_combo.addItems(["0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(2)  # 1.0x default
        self.speed_combo.setEnabled(False)
        self.speed_combo.setToolTip("Playback Speed")
        
        # Erstelle einen dunkleren SVG-Pfeil für den deaktivierten Zustand
        self.create_disabled_arrow_svg()
        
        # Set down arrow icon with different styles for enabled/disabled states
        self.speed_combo.setStyleSheet(self.speed_combo.styleSheet() + f"""
            QComboBox::down-arrow {{
                width: 16px;
                height: 16px;
            }}
        """)
        
        # Setze Pfeil basierend auf aktuellem Zustand
        self.update_dropdown_arrow(False)
        
        self.speed_combo.currentIndexChanged.connect(self.speed_changed)
        media_controls_layout.addWidget(self.speed_combo)
        
        time_controls_layout.addLayout(media_controls_layout)
        time_controls_layout.addStretch(1)
        
        # Right controls group
        right_controls_layout = QHBoxLayout()
        right_controls_layout.setSpacing(15)
        
        # Download button
        self.download_btn = create_svg_icon_button(
            os.path.join(os.path.dirname(__file__), "images/download.svg"),
            (20, 20),
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
    
    def create_disabled_arrow_svg(self):
        """Erstellt eine dunklere Version des Dropdown-Pfeils für den deaktivierten Zustand."""
        # Pfade für die SVG-Dateien
        self.down_arrow_path = os.path.join(os.path.dirname(__file__), "down.svg")
        self.down_arrow_disabled_path = os.path.join(os.path.dirname(__file__), "down_disabled.svg")
        
        # Erstellt die deaktivierte Version, wenn sie noch nicht existiert
        if not os.path.exists(self.down_arrow_disabled_path):
            try:
                with open(self.down_arrow_path, 'r') as file:
                    svg_content = file.read()
                
                # Ersetze die weiße Farbe durch eine dunklere Farbe mit Transparenz
                disabled_svg = svg_content.replace('fill="#ffffff"', 'fill="#ffffff" opacity="0.4"')
                
                with open(self.down_arrow_disabled_path, 'w') as file:
                    file.write(disabled_svg)
            except Exception as e:
                print(f"Fehler beim Erstellen der deaktivierten SVG: {e}")
                # Fallback: Verwenden Sie das normale SVG für beide Zustände
                self.down_arrow_disabled_path = self.down_arrow_path
    
    def update_dropdown_arrow(self, enabled):
        """Aktualisiert das Dropdown-Pfeil-Icon basierend auf dem Aktivierungsstatus."""
        arrow_path = self.down_arrow_path if enabled else self.down_arrow_disabled_path
        self.speed_combo.setStyleSheet(self.speed_combo.styleSheet().split("QComboBox::down-arrow {")[0] + f"""
            QComboBox::down-arrow {{
                image: url("{arrow_path}");
                width: 16px;
                height: 16px;
            }}
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
            
            # Enable buttons and controls
            self.play_btn.setEnabled(True)
            self.rewind_btn.setEnabled(True)
            self.forward_btn.setEnabled(True)
            self.speed_combo.setEnabled(True)
            self.update_dropdown_arrow(True)  # Aktualisiere den Dropdown-Pfeil
            self.position_slider.setEnabled(True)
            self.download_btn.setEnabled(True)
            self.time_label.setProperty("enabled", True)
            self.time_label.style().unpolish(self.time_label)
            self.time_label.style().polish(self.time_label)
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
        # Disable all controls
        self.play_btn.setEnabled(False)
        self.rewind_btn.setEnabled(False)
        self.forward_btn.setEnabled(False)
        self.speed_combo.setEnabled(False)
        self.update_dropdown_arrow(False)  # Aktualisiere den Dropdown-Pfeil
        self.position_slider.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.time_label.setProperty("enabled", False)
        self.time_label.style().unpolish(self.time_label)
        self.time_label.style().polish(self.time_label)
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