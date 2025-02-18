# Verwende ein Basis-Image mit Miniconda
FROM continuumio/miniconda3:latest

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die environment.yml ins Image
COPY environment.yml .

# Erstelle das Conda-Environment aus der YAML-Datei
RUN conda env create -f environment.yml

# Sorge dafür, dass das neu erstellte Environment verwendet wird
ENV PATH=/opt/conda/envs/bmc/bin:$PATH

# Kopiere den gesamten Inhalt deines Repositories in das Container-Verzeichnis
COPY . /app

# Installiere dein Paket im "editable" Modus (setup.py wird genutzt)
RUN pip install -e .

# Führe beim Start ein bestimmtes Python-Skript aus (ändere 'dein_script.py' falls nötig)
# CMD ["python", "dein_script.py"]