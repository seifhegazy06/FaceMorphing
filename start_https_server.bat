@echo off
echo Starting HTTPS server for mobile camera access...
echo.
echo First, generating SSL certificate...
python generate_cert.py
echo.
echo Starting server on HTTPS (port 8000)...
echo Access from mobile: https://192.168.137.24:8000
echo.
echo IMPORTANT: Your browser will show a security warning.
echo Click "Advanced" and then "Proceed anyway" or "Accept Risk"
echo.
uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem --reload
pause
