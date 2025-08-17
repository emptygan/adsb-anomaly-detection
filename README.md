# ADS-B Anomaly Detection – README

## Project Layout

├─ time_splits/                  # Raw ADS-B data, split hourly (hour_07_data.csv ...)

├─ all_aircraft_db.csv           # ICAO24 reference database

├─ jet_only_aircraft_opensky.csv # Commercial jet whitelist

├─ consolidated_flight_data.csv  # Cleaned & merged dataset

├─ flight_summary.csv            # Extracted flight features

├─ flight_summary_with_iso.csv   # Isolation Forest + Kneedle results

├─ flight_summary_kmeans.csv     # PCA + KMeans clustering results

├─ final_consensus_anomalies.csv # Consolidated anomaly results

│

├─ aircraft_filter.py

├─ merge_icao24_db.py

├─ flight_summary_generator.py

├─ z_score.py

├─ iso_kneed.py

├─ k_means.py

├─ flight_path.py

├─ merge.py

├─ vis.py / vis_pro.py / cluster_and_plot.py

└─ run_all_detection.py          # Suppose to run all steps in sequence, might not work


## Environment
- Python 3.9–3.11
- Required libraries:  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `kneed`

Install quickly:

pip install pandas numpy scikit-learn matplotlib kneed


## Workflow
Typical workflow:

1. **Data Cleaning & Merging**  
   python merge_icao24_db.py --input ./time_splits/hour_07_data.csv                              --aircraft-db ./all_aircraft_db.csv                              --output ./consolidated_flight_data.csv
   python aircraft_filter.py --input ./consolidated_flight_data.csv                              --jet-whitelist ./jet_only_aircraft_opensky.csv                              --output ./consolidated_flight_data.csv


2. **Feature Extraction**  
   python flight_summary_generator.py           --input ./consolidated_flight_data.csv           --output ./flight_summary.csv


3. **Model Execution**  
   - Z-score: `z_score.py --z 3.5`  
   - Isolation Forest + Kneedle: `iso_kneed.py --max-trees 200`  
   - PCA + KMeans: `k_means.py --k 3`  
   - Turning-angle rule: `flight_path.py --angle 80 --window 180`  

4. **Result Integration & Visualization**  
   python merge.py --output ./final_consensus_anomalies.csv
   python vis.py --input ./final_consensus_anomalies.csv


Or run everything at once:
python run_all_detection.py --raw-dir ./time_splits                             --aircraft-db ./all_aircraft_db.csv                             --jet-whitelist ./jet_only_aircraft_opensky.csv                             --out-dir .

## Notes
- **Absolute paths**: Original experiment scripts used absolute local paths. This version has been updated to use **relative paths and CLI arguments** for reproducibility.  
- **Hourly split**: Raw data is stored per hour (07–10) to avoid memory overflow.  
- **Filtering**: Uses `all_aircraft_db.csv` and `jet_only_aircraft_opensky.csv` to filter out non-commercial jets.  
- **Visualization**: Outputs include clustering scatter plots, histograms, anomaly detection comparisons, and trajectory maps.  
- **RawData**: All raw data has been removed since they are too big. Including ICAO24 call sign database.
