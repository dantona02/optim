import os
import sys
from pathlib import Path
import numpy as np
import torch
import time
import uuid
from manim import *
from manim.opengl import *
from PyQt6.QtCore import QObject, pyqtSignal

# Unterdrücke Manim-Warnungen
import logging
logging.getLogger('manim').setLevel(logging.ERROR)


class AnimationProgressListener(QObject):
    """Klasse zum Übermitteln von Fortschrittsupdates für die Animation"""
    progress_updated = pyqtSignal(int, int)  # (current, total)
    status_updated = pyqtSignal(str)


class AnimationAdapter:
    """Adapter-Klasse für die BMCSim-Animation, die ohne Jupyter funktioniert."""
    
    def __init__(self, progress_listener=None):
        """
        Initialisiert den Animation Adapter.
        
        Parameters
        ----------
        progress_listener : AnimationProgressListener, optional
            Ein Listener für Fortschrittsupdates
        """
        self.progress_listener = progress_listener
        
    def animate(self, sim_engine, animation_params):
        """
        Erstellt eine Animation aus einem sim_engine-Objekt mit den angegebenen Parametern.
        
        Parameters
        ----------
        sim_engine : BMCSim
            Das BMCSim-Objekt mit den Simulationsdaten
        animation_params : dict
            Ein Dictionary mit den Animation-Parametern
            
        Returns
        -------
        str or None
            Der Pfad zur erstellten Videodatei, oder None bei Fehler
        """
        try:
            if self.progress_listener:
                self.progress_listener.status_updated.emit("Preparing animation...")
            
            # Extrahiere Parameter
            step = animation_params.get("step", 1)
            run_time = animation_params.get("run_time", 0.1)
            track_path = animation_params.get("track_path", True)
            timing = animation_params.get("timing", False)
            total_mag = animation_params.get("total_mag", False)
            animate_cest = animation_params.get("animate_cest", True)
            quality_param = animation_params.get("quality", "low") # low, medium, high
            
            # Stelle sicher, dass wir mit CPU-Tensoren arbeiten
            time_array = sim_engine.t[::step].cpu()
            m_out = sim_engine.m_out.cpu()
            total_vec = sim_engine.total_vec.cpu() if hasattr(sim_engine, "total_vec") and sim_engine.total_vec is not None else None
            z_positions = sim_engine.z_positions.cpu()
            isochromats = sim_engine.n_isochromats
            
            # Magnetisierung für Wasserpools vorbereiten
            if animate_cest:
                n_total_pools = len(sim_engine.params.cest_pools) + 1
                m_vec_water = np.stack(
                    (m_out[:, 1, :],
                    m_out[:, n_total_pools + 1, :],
                    m_out[:, sim_engine.params.mz_loc + 1, :]),
                    axis=2
                )
                # Normiere jeden Vektor in m_vec_water auf Länge 1
                norms = np.linalg.norm(m_vec_water, axis=2, keepdims=True)
                norms[norms == 0] = 1  # Division durch 0 vermeiden
                m_vec_water = m_vec_water / norms
            else:
                if sim_engine.params.cest_pools is not None:
                    n_total_pools = len(sim_engine.params.cest_pools) + 1
                    m_vec_water = np.stack(
                        (m_out[:, 0, :],
                        m_out[:, n_total_pools, :],
                        m_out[:, sim_engine.params.mz_loc, :]),
                        axis=2)
                else:
                    m_vec_water = np.stack(
                        (m_out[:, 0, :],
                        m_out[:, 1, :],
                        m_out[:, sim_engine.params.mz_loc, :]),
                        axis=2)
            
            if total_mag and total_vec is not None:
                m_vec_total = np.stack(
                    (total_vec[:, 0],
                    total_vec[:, 1],
                    total_vec[:, 2]),
                    axis=1
                )
            else:
                m_vec_total = None

            m_vec_water = m_vec_water[:, ::step]
            if m_vec_total is not None:
                m_vec_total = m_vec_total[::step]  # Schrittweite anwenden
            middle_idx = np.where(z_positions == 0)[0][0]
            m_vec_middle = m_vec_water[middle_idx] if total_mag else None
            
            # Erstelle Ausgabeverzeichnisse
            media_dir = Path(__file__).resolve().parent.parent.parent / "media"
            output_dir = media_dir / "custom_animations"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Generiere eine eindeutige ID für diese Animation
            unique_id = str(uuid.uuid4())[:8]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"animation_{unique_id}_{timestamp}.mp4"
            output_path = output_dir / output_filename
            
            if self.progress_listener:
                self.progress_listener.status_updated.emit("Creating animation...")
            
            # Konfiguriere manim global
            config.output_file = output_filename
            config.video_dir = str(output_dir)
            config.write_to_movie = True
            config.disable_caching = True
            config.frame_rate = 30
            
            if quality_param == "high":
                config.quality = "high_quality"
                config.pixel_width = 1920
                config.pixel_height = 1080
            elif quality_param == "medium":
                config.quality = "medium_quality"
                config.pixel_width = 1280
                config.pixel_height = 720
            else:
                config.quality = "low_quality"
                config.pixel_width = 854
                config.pixel_height = 480
                
            # Erstelle eine MagnetisierungAnimationScene-Klasse mit den Daten
            class MagnetisationAnimationScene(ThreeDScene):
                def __init__(self, *args, **kwargs):
                    self.time_array = time_array
                    self.m_vec_water = m_vec_water
                    self.m_vec_total = m_vec_total
                    self.m_vec_middle = m_vec_middle
                    self.isochromats = isochromats
                    self.run_time_per_step = run_time
                    self.track_path = track_path
                    self.timing = timing
                    self.total_mag = total_mag
                    self.progress_listener = kwargs.pop('progress_listener', None)
                    super().__init__(*args, **kwargs)

                def construct(self):
                    # Achsen und Skalierungsfaktoren
                    axes = ThreeDAxes(
                        x_range=(-1.4, 1.4, .2),
                        y_range=(-1.4, 1.4, .2),
                        z_range=(-1.4, 1.4, .2)
                    )
                    scale_factor_xy = axes.x_length / (axes.x_range[1] - axes.x_range[0])
                    scale_factor_z = axes.z_length / (axes.z_range[1] - axes.z_range[0])
                    scaling_array = np.array([scale_factor_xy, scale_factor_xy, scale_factor_z])
                    labels = axes.get_axis_labels(Text("x").scale(.7), Text("y").scale(.7), Text("z").scale(.7))
                    
                    # Farbübergang für Isochromaten
                    colors = color_gradient([PURE_BLUE, PURE_GREEN, PURE_RED], self.isochromats)

                    # Tracker und Vektoren für alle Isochromaten initialisieren
                    trackers = []
                    vectors = []
                    paths = []
                    
                    total_frames = len(self.time_array)

                    if self.total_mag and self.m_vec_total is not None and self.m_vec_middle is not None:
                        for i, vec in enumerate([self.m_vec_total, self.m_vec_middle]):
                            colors_total_mag = [WHITE, PURE_GREEN]
                            x_tracker = ValueTracker(vec[0, 0] * scaling_array[0])
                            y_tracker = ValueTracker(vec[0, 1] * scaling_array[1])
                            z_tracker = ValueTracker(vec[0, 2] * scaling_array[2])
                            trackers.append((x_tracker, y_tracker, z_tracker))

                            # Vektor erstellen
                            vector = Vector(
                                [x_tracker.get_value(), y_tracker.get_value(), z_tracker.get_value()],
                                color=colors_total_mag[i]
                            )

                            def create_updater(x, y, z, col):
                                def update_vector(v):
                                    v.become(Vector([x.get_value(), y.get_value(), z.get_value()], color=col))
                                return update_vector

                            vector.add_updater(create_updater(x_tracker, y_tracker, z_tracker, colors_total_mag[i]))
                            vectors.append(vector)

                            # Pfad hinzufügen, falls aktiviert
                            if self.track_path:
                                path = TracedPath(vector.get_end, stroke_color=colors_total_mag[i], stroke_width=1)
                                paths.append(path)

                        self.add(axes, labels, *vectors)
                        if self.track_path:
                            self.add(*paths)

                        self.set_camera_orientation(phi=65 * DEGREES, theta=135 * DEGREES)

                        # Text für die Zeit
                        if self.timing:
                            decimal = Text("0", font_size=36)
                            time_tracker = ValueTracker(0)

                            def update_decimal(d):
                                current_index = int(time_tracker.get_value())
                                current_time = self.time_array[current_index] if current_index < len(self.time_array) else self.time_array[-1]
                                d.become(Text(f"t = {current_time:.4f} s", font_size=36))
                                d.fix_in_frame()
                                d.to_corner(UR).scale(0.7)

                            decimal.add_updater(update_decimal)
                            self.add(decimal)

                        # Animation über die Zeit
                        for t in range(1, len(self.time_array)):
                            # Update progress if listener available
                            if self.progress_listener:
                                self.progress_listener.progress_updated.emit(t, total_frames)
                                self.progress_listener.status_updated.emit(f"Creating animation... Frame {t}/{total_frames}")
                            
                            self.play(
                                *[
                                    trackers[i][0].animate.set_value(vec[t, 0] * scaling_array[0]) for i, vec in enumerate([self.m_vec_total, self.m_vec_middle])
                                ] + [
                                    trackers[i][1].animate.set_value(vec[t, 1] * scaling_array[1]) for i, vec in enumerate([self.m_vec_total, self.m_vec_middle])
                                ] + [
                                    trackers[i][2].animate.set_value(vec[t, 2] * scaling_array[2]) for i, vec in enumerate([self.m_vec_total, self.m_vec_middle])
                                ] + (
                                    [time_tracker.animate.set_value(t)] if self.timing else []
                                ),
                                run_time=self.run_time_per_step, rate_func=linear
                            )

                    else:
                        for i in range(self.isochromats):
                            x_tracker = ValueTracker(self.m_vec_water[i, 0, 0] * scaling_array[0])
                            y_tracker = ValueTracker(self.m_vec_water[i, 0, 1] * scaling_array[1])
                            z_tracker = ValueTracker(self.m_vec_water[i, 0, 2] * scaling_array[2])
                            trackers.append((x_tracker, y_tracker, z_tracker))

                            # Vektor erstellen
                            vector = Vector(
                                [x_tracker.get_value(), y_tracker.get_value(), z_tracker.get_value()],
                                color=colors[i]
                            )

                            def create_updater(x, y, z, col):
                                def update_vector(v):
                                    v.become(Vector([x.get_value(), y.get_value(), z.get_value()], color=col))
                                return update_vector

                            vector.add_updater(create_updater(x_tracker, y_tracker, z_tracker, colors[i]))
                            vectors.append(vector)

                            # Pfad hinzufügen, falls aktiviert
                            if self.track_path:
                                path = TracedPath(vector.get_end, stroke_color=colors[i], stroke_width=1)
                                paths.append(path)

                        self.add(axes, labels, *vectors)
                        if self.track_path:
                            self.add(*paths)

                        self.set_camera_orientation(phi=65 * DEGREES, theta=135 * DEGREES)

                        # Text für die Zeit
                        if self.timing:
                            decimal = Text("0", font_size=36)
                            time_tracker = ValueTracker(0)

                            def update_decimal(d):
                                current_index = int(time_tracker.get_value())
                                current_time = self.time_array[current_index] if current_index < len(self.time_array) else self.time_array[-1]
                                d.become(Text(f"t = {current_time:.4f} s", font_size=36))
                                d.fix_in_frame()
                                d.to_corner(UR).scale(0.7)

                            decimal.add_updater(update_decimal)
                            self.add(decimal)

                        # Animation über die Zeit
                        for t in range(1, len(self.time_array)):
                            # Update progress if listener available
                            if self.progress_listener:
                                self.progress_listener.progress_updated.emit(t, total_frames)
                                self.progress_listener.status_updated.emit(f"Creating animation... Frame {t}/{total_frames}")
                                
                            self.play(
                                *[
                                    trackers[i][0].animate.set_value(self.m_vec_water[i, t, 0] * scaling_array[0]) for i in range(self.isochromats)
                                ] + [
                                    trackers[i][1].animate.set_value(self.m_vec_water[i, t, 1] * scaling_array[1]) for i in range(self.isochromats)
                                ] + [
                                    trackers[i][2].animate.set_value(self.m_vec_water[i, t, 2] * scaling_array[2]) for i in range(self.isochromats)
                                ] + (
                                    [time_tracker.animate.set_value(t)] if self.timing else []
                                ),
                                run_time=self.run_time_per_step, rate_func=linear
                            )
            
            # Rendere die Szene direkt
            scene = MagnetisationAnimationScene(progress_listener=self.progress_listener)
            scene.render()
            
            # Prüfe, ob die Datei erstellt wurde
            if output_path.exists():
                if self.progress_listener:
                    self.progress_listener.status_updated.emit("Animation completed successfully")
                return str(output_path)
            else:
                # Suche nach dem erstellten Video im Ausgabeverzeichnis
                for file in output_dir.glob("*.mp4"):
                    if file.stat().st_mtime > time.time() - 60:  # Innerhalb der letzten Minute erstellt
                        if self.progress_listener:
                            self.progress_listener.status_updated.emit("Animation completed successfully")
                        return str(file)
            
            # Keine Datei gefunden
            if self.progress_listener:
                self.progress_listener.status_updated.emit("Could not locate the generated video")
            return None
            
        except Exception as e:
            if self.progress_listener:
                self.progress_listener.status_updated.emit(f"Error in animation: {str(e)}")
            print(f"Animation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None