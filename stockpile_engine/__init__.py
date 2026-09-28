"""Stockpile Intelligence Engine — volumetric + thermal monitoring of bulk stockpiles.

Sister engine of the Solain Thermal Engine. Same deployment shape (Flask on
Cloud Run, Pub/Sub push, GCS outputs, optional BigQuery), same credentials
(GCP_CREDENTIALS_GEE / GCP_CREDENTIALS_GCS), different question: how much
material is on the yard, what is it, and is any of it heating up.
"""

__version__ = "1.0.0"
