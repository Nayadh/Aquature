@echo off
echo ============================================
echo  Water Quality Pipeline - Auto Run
echo ============================================
echo.

call venv\Scripts\activate.bat

echo [1/2] Generating labels...
python src\auto_labels.py

echo.
echo [2/2] Running full analysis...
python main.py ^
  --scene data\raw\tanager\20250511_074311_00_4001_ortho_sr_hdf5.h5 ^
  --stac data\raw\tanager\20250511_074311_00_4001_stac.json ^
  --labels data\labels\labels_auto.csv ^
  --outdir outputs\results ^
  --epochs 50

echo.
echo ============================================
echo  DONE. Open index.html to view results.
echo ============================================
pause