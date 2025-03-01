from PyQt6.QtCore import QThread, pyqtSignal
import os
import glob
import re
import time
from pathlib import Path
from datetime import datetime
from bmc.gui.animation_adapter import AnimationAdapter, AnimationProgressListener


class AnimationWorker(QThread):
    """Worker thread for creating animations without freezing the UI."""
    progress_updated = pyqtSignal(int, int)  # (current, total)
    status_updated = pyqtSignal(str)
    animation_completed = pyqtSignal(str)  # Sends the video path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, sim_engine, animation_params):
        super().__init__()
        self.sim_engine = sim_engine
        self.animation_params = animation_params
        self.creation_time = None
        
    def run(self):
        """Run the animation in a separate thread."""
        try:
            self.status_updated.emit("Starting animation creation...")
            
            # Estimate total frames for progress calculation
            total_frames = len(self.sim_engine.t[::self.animation_params['step']])
            self.progress_updated.emit(0, total_frames)
            
            # Create progress listener for Animation Adapter
            progress_listener = AnimationProgressListener()
            progress_listener.progress_updated.connect(self.progress_updated)
            progress_listener.status_updated.connect(self.status_updated)
            
            # Use Animation Adapter instead of direct animate call
            adapter = AnimationAdapter(progress_listener)
            
            # Record creation time to help find the correct video
            self.creation_time = time.time()
            
            # Run animation
            video_path = adapter.animate(self.sim_engine, self.animation_params)
            
            if video_path and os.path.exists(video_path):
                self.animation_completed.emit(video_path)
                self.status_updated.emit("Animation completed successfully")
            else:
                self.error_occurred.emit("Could not locate the generated video")
            
        except Exception as e:
            self.error_occurred.emit(f"Error during animation: {str(e)}")
    
    def _find_latest_video(self):
        """Find the latest generated video file that was created after the animation started."""
        try:
            # Get the media directory from the current file
            media_dir = Path(__file__).resolve().parent.parent.parent / "media" / "jupyter"
            if not media_dir.exists():
                # Also check the videos directory as a fallback
                media_dir = Path(__file__).resolve().parent.parent.parent / "media" / "videos"
                if not media_dir.exists():
                    self.error_occurred.emit(f"Media directory not found: {media_dir}")
                    return None
            
            # First try the jupyter media directory which is the default location
            mp4_files = []
            
            # Check jupyter directory first (more likely to contain the animation)
            jupyter_dir = Path(__file__).resolve().parent.parent.parent / "media" / "jupyter"
            if jupyter_dir.exists():
                mp4_files = list(jupyter_dir.glob("**/*.mp4"))
            
            # If no files found in jupyter dir, try videos directory
            if not mp4_files:
                # Find all Vector3DScene directories in videos dir
                scene_dirs = list(media_dir.glob("**/Vector3DScene"))
                if scene_dirs:
                    # Sort by creation time (newest last)
                    scene_dirs.sort(key=os.path.getmtime)
                    latest_dir = scene_dirs[-1]
                    
                    # Find all mp4 files in the latest directory
                    mp4_files = list(latest_dir.glob("**/*.mp4"))
            
            # If we still don't have any files, search more broadly in the media directory
            if not mp4_files:
                mp4_files = list(media_dir.glob("**/*.mp4"))
            
            # Filter for only files created after the animation started
            if self.creation_time and mp4_files:
                mp4_files = [f for f in mp4_files if os.path.getmtime(f) >= self.creation_time]
            
            # Sort by creation time (newest last) and return the newest
            if mp4_files:
                mp4_files.sort(key=os.path.getmtime)
                return str(mp4_files[-1])
            
            return None
            
        except Exception as e:
            self.error_occurred.emit(f"Error finding video: {str(e)}")
            return None