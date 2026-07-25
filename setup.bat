@echo off
echo Creando entorno virtual...
python -m venv venv
call venv\Scripts\activate
echo Instalando dependencias...
pip install -r requirements.txt
echo Entorno listo.
pause
